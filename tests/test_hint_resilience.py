"""Transport regressions only: no network, credentials, CUDA or loss changes."""
import io
import json
import socket
import ssl
import threading
import urllib.error
import urllib.request
from collections import Counter
import pytest
from rethink_webshop import common, hint_api, hints


PRIVATE_DETAIL='sensitive provider or filesystem detail'


@pytest.fixture
def episode():
    return {'task':'Find a blue cotton shirt','steps':[
        {'step':0,'triggered':True,'observation':'Search page',
         'reasoning':'Search for the requested item.','action':'search[shirt]',
         'feedback':'Results','available_actions':{'has_search_bar':True,'clickables':[]}}]}


@pytest.fixture
def configured(monkeypatch):
    cfg={'base':'https://provider.example/v1','key':'test-placeholder','model':hint_api.MODEL}
    monkeypatch.setattr(hint_api,'auth',lambda:cfg)
    return cfg


def response():
    return io.BytesIO(json.dumps({'choices':[{'finish_reason':'stop','message':{
        'content':json.dumps({'hints':{'0':"Next command: 'search[shirt]'. Search."}})}}]}).encode())


def batch(episodes):
    class Batch:pass
    item=Batch()
    item.non_tensor_batch={'rethink_episode_json':[json.dumps(e) for e in episodes],
                           'rethink_step':[0]*len(episodes),'rethink_triggered':[True]*len(episodes)}
    return item


def test_cache_write_failure_is_fatal_and_never_repeats_http(tmp_path,monkeypatch,configured,episode,capsys):
    calls=[];sleeps=[]
    def opener(*args,**kwargs):calls.append(1);return response()
    def failed(*args,**kwargs):raise PermissionError(PRIVATE_DETAIL)
    monkeypatch.setattr(hint_api,'atomic_json',failed)
    with pytest.raises(hint_api.HintAPIFatal,match='api_cache_write_error') as caught:
        hint_api.fetch(episode,tmp_path,opener=opener,sleeper=sleeps.append)
    assert len(calls)==1 and not sleeps
    assert caught.value.fatal and caught.value.diagnostics['api_response_received']
    assert caught.value.diagnostics['stage']=='cache_write'
    assert PRIVATE_DETAIL not in capsys.readouterr().out+str(caught.value)


def test_corrupt_cache_does_not_call_api_again(tmp_path,configured,episode):
    hint_api.fetch(episode,tmp_path,opener=lambda *a,**kw:response())
    next(tmp_path.glob('*.json')).write_text('{broken',encoding='utf-8')
    def forbidden(*args,**kwargs):raise AssertionError('Must not repeat the paid request')
    with pytest.raises(hint_api.HintAPIFatal,match='api_cache_read_error'):
        hint_api.fetch(episode,tmp_path,opener=forbidden)


@pytest.mark.parametrize('stage',['open','read'])
def test_timeout_stage_is_preserved_without_private_text(stage,tmp_path,configured,episode,capsys):
    class Slow(io.BytesIO):
        def read(self,*args):raise TimeoutError(PRIVATE_DETAIL)
    def opener(*args,**kwargs):
        if stage=='open':raise TimeoutError(PRIVATE_DETAIL)
        return Slow()
    with pytest.raises(hint_api.HintAPIHold) as caught:
        hint_api.fetch(episode,tmp_path,opener=opener,retries=0)
    assert caught.value.temporary and not caught.value.fatal
    assert caught.value.diagnostics['stage']==stage
    assert caught.value.diagnostics['error_kind']=='timeout'
    assert PRIVATE_DETAIL not in capsys.readouterr().out+str(caught.value)


def test_dns_error_is_not_a_generic_undifferentiated_hold(tmp_path,configured,episode):
    def failed(*args,**kwargs):raise urllib.error.URLError(socket.gaierror(-2,PRIVATE_DETAIL))
    with pytest.raises(hint_api.HintAPIHold) as caught:
        hint_api.fetch(episode,tmp_path,opener=failed,retries=0)
    assert caught.value.diagnostics['error_kind']=='dns_error'
    assert caught.value.diagnostics['stage']=='open'


def test_certificate_failure_stops_without_retries(tmp_path,configured,episode):
    calls=[]
    def failed(*args,**kwargs):
        calls.append(1)
        raise urllib.error.URLError(ssl.SSLCertVerificationError(1,PRIVATE_DETAIL))
    with pytest.raises(hint_api.HintAPIFatal,match='api_tls_certificate_error'):
        hint_api.fetch(episode,tmp_path,opener=failed,sleeper=lambda _:pytest.fail('No retry'))
    assert len(calls)==1


@pytest.mark.parametrize('code',[400,401,403,404])
def test_permanent_http_errors_are_actionable_fatal(code,tmp_path,configured,episode):
    calls=[]
    def failed(*args,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://provider.example/v1',code,PRIVATE_DETAIL,{},None)
    with pytest.raises(hint_api.HintAPIFatal,match='http_'+str(code)) as caught:
        hint_api.fetch(episode,tmp_path,opener=failed,sleeper=lambda _:pytest.fail('No retry'))
    assert len(calls)==1 and caught.value.diagnostics['http_status']==code


def test_transient_http_retry_budget_is_unchanged(tmp_path,configured,episode):
    calls=[];sleeps=[]
    def failed(*args,**kwargs):
        calls.append(1)
        raise urllib.error.HTTPError('https://provider.example/v1',429,PRIVATE_DETAIL,{},None)
    with pytest.raises(hint_api.HintAPIHold) as caught:
        hint_api.fetch(episode,tmp_path,opener=failed,sleeper=sleeps.append)
    assert len(calls)==6 and sleeps==[2,4,8,16,32]
    assert caught.value.temporary and not caught.value.fatal


def test_permanent_http_error_does_not_enter_outer_infinite_retry(tmp_path,monkeypatch,episode):
    calls=[]
    def failed(*args,**kwargs):calls.append(1);raise hint_api.HintAPIFatal('http_401')
    monkeypatch.setattr(hints,'fetch',failed)
    monkeypatch.setattr(hints.time,'sleep',lambda _:pytest.fail('Permanent error must stop'))
    with pytest.raises(hint_api.HintAPIFatal,match='http_401'):
        hints.collect_hints(batch([episode]),tmp_path)
    assert len(calls)==1
    status=json.loads((tmp_path/'api_wait_status.json').read_text())
    assert status['status']=='failed' and status['reason']=='http_401'
    assert status['same_batch_retained'] and not status['actor_updated']


def test_successful_episodes_survive_another_episode_retry(tmp_path,monkeypatch,episode):
    other={**episode,'task':'retry'};first_done=threading.Event();calls=Counter();sleeps=[]
    def request(ep,*args,**kwargs):
        calls[ep['task']]+=1
        if ep['task']=='retry' and calls[ep['task']]==1:
            assert first_done.wait(2)
            raise hint_api.HintAPIHold('api_transport_error',temporary=True)
        first_done.set()
        return {0:'hint for '+ep['task']},{}
    monkeypatch.setattr(hints,'fetch',request)
    monkeypatch.setattr(hints.time,'sleep',sleeps.append)
    values,reasons=hints.collect_hints(batch([episode,other]),tmp_path,workers=2)
    assert values==['hint for '+episode['task'],'hint for retry'] and reasons==['applied','applied']
    assert calls[episode['task']]==1 and calls['retry']==2 and sleeps==[60]
    assert json.loads((tmp_path/'api_wait_status.json').read_text())['status']=='ready'


def test_unwritable_status_directory_fails_before_paid_call(tmp_path,monkeypatch,episode):
    def forbidden(*args,**kwargs):pytest.fail('Disk check must fail before any request')
    def failed(*args,**kwargs):raise PermissionError(PRIVATE_DETAIL)
    monkeypatch.setattr(hints,'fetch',forbidden)
    monkeypatch.setattr(hints,'atomic_json',failed)
    with pytest.raises(hint_api.HintAPIFatal,match='api_status_write_error'):
        hints.collect_hints(batch([episode]),tmp_path)


def test_batch_emits_heartbeat_while_waiting(tmp_path,monkeypatch,episode):
    original=hints.wait;states=[];calls=[]
    def wait_once(futures,timeout,return_when):
        calls.append(timeout)
        if len(calls)==1:return set(),set(futures)
        return original(futures,timeout=timeout,return_when=return_when)
    writer=hints.atomic_json
    def status(path,value):states.append(value['status']);writer(path,value)
    monkeypatch.setattr(hints,'wait',wait_once)
    monkeypatch.setattr(hints,'atomic_json',status)
    monkeypatch.setattr(hints,'fetch',lambda *a,**kw:({0:'hint'},{}))
    assert hints.collect_hints(batch([episode]),tmp_path)==(['hint'],['applied'])
    assert 'waiting_requests' in states and states[-1]=='ready' and calls[0]==30


@pytest.mark.parametrize('base',[
    'http://provider.example/v1','https://provider.example/v1/chat/completions',
    'https://provider.example/v1/chat/completions/','https://user:secret@provider.example/v1',
    'https://provider.example/v1?key=not-a-real-key','https://provider.example:bad/v1'])
def test_wrong_base_is_rejected_before_network(base,configured):
    with pytest.raises(ValueError):hint_api.validate_auth({**configured,'base':base})


def test_environment_proxy_is_explicitly_switchable(monkeypatch):
    monkeypatch.setattr(urllib.request,'getproxies',lambda:{'https':'http://127.0.0.1:9'})
    monkeypatch.setenv('RETHINK_HINT_TRUST_ENV','true')
    opener=hint_api.transport_opener().__self__
    assert any(isinstance(h,urllib.request.ProxyHandler) and h.proxies for h in opener.handlers)
    monkeypatch.setenv('RETHINK_HINT_TRUST_ENV','false')
    direct=hint_api.transport_opener().__self__
    assert not any(isinstance(h,urllib.request.ProxyHandler) and h.proxies for h in direct.handlers)
    monkeypatch.setenv('RETHINK_HINT_TRUST_ENV','invalid')
    with pytest.raises(hint_api.HintAPIFatal,match='api_invalid_proxy_option'):
        hint_api.transport_opener()


def test_cli_local_checks_never_call_network(tmp_path,monkeypatch,configured,capsys):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.delenv('RETHINK_HINT_AUTH',raising=False)
    for name,value in [('HINT_API_BASE',configured['base']),('HINT_API_KEY',configured['key']),('HINT_MODEL',configured['model'])]:
        monkeypatch.setenv(name,value)
    monkeypatch.setattr(hint_api,'transport_opener',lambda:pytest.fail('No API opt-in'))
    assert hint_api.main(['--check'])==0
    assert configured['key'] not in capsys.readouterr().out


def test_cli_paid_checks_use_fresh_cache_and_exact_parser(tmp_path,monkeypatch,configured):
    monkeypatch.setattr(common,'ROOT',tmp_path)
    monkeypatch.delenv('RETHINK_HINT_AUTH',raising=False)
    for name,value in [('HINT_API_BASE',configured['base']),('HINT_API_KEY',configured['key']),('HINT_MODEL',configured['model'])]:
        monkeypatch.setenv(name,value)
    calls=[]
    def opener(req,**kwargs):
        calls.append(json.loads(req.data))
        return response()
    monkeypatch.setattr(hint_api,'transport_opener',lambda:opener)
    assert hint_api.main(['--check','--allow-api-cost'])==0
    assert hint_api.main(['--check','--allow-api-cost'])==0
    assert len(calls)==2 and calls[0]==calls[1]
    assert calls[0]['model']==hint_api.MODEL and calls[0]['reasoning_effort']=='low'
