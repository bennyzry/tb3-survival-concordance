#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Reproduce one independent standard trial; secrets never enter argv/configs."""
from pathlib import Path
import argparse, hashlib, json, os, subprocess

p=argparse.ArgumentParser();p.add_argument('provider',choices=['codex','deepseek']);p.add_argument('--id',required=True)
p.add_argument('--agent-directory',type=Path,default=Path.home()/'.cache/tb3-survival-agents')
p.add_argument('--print-config',action='store_true');args=p.parse_args()
assert args.id and all(c.isalnum() or c in '-_' for c in args.id),'Use a simple unique trial ID'
repo=Path(__file__).resolve().parents[1]
runtime=repo/'jobs';runtime.mkdir(exist_ok=True)
configpath=runtime/(args.id+'.json');assert not configpath.exists(),'Choose a new ID; do not overwrite evidence'
agents=json.loads((args.agent_directory/'agents.json').read_text(encoding='utf-8'))
name='codex' if args.provider=='codex' else 'claude'
agent=next(row for row in agents if row['name']==name)
binary=Path(agent['executable']);assert hashlib.sha256(binary.read_bytes()).hexdigest()==agent['sha256']
config={'job_name':args.id,'jobs_dir':str(runtime),'n_attempts':1,'n_concurrent_trials':1,'quiet':True,
    'retry':{'max_retries':0},'tasks':[{'path':str(repo/'tasks/survival-concordance')}],
    'agents':[{'name':'codex' if name=='codex' else 'claude-code','model_name':'openai/gpt-6.1-sol' if name=='codex' else 'deepseek-flash',
        'kwargs':{'version':agent['version'],'reasoning_effort':'xhigh' if name=='codex' else 'max'}}],
    'environment':{'type':'docker','delete':True,'stream':False,'mounts':[{'type':'bind','source':str(binary),
        'target':'/usr/local/bin/'+name,'read_only':True}]}}
env=dict(os.environ,LITELLM_LOCAL_MODEL_COST_MAP='True')
for key in ('OPENAI_API_KEY','OPENAI_BASE_URL','ANTHROPIC_API_KEY','ANTHROPIC_AUTH_TOKEN','CLAUDE_CODE_OAUTH_TOKEN'):env.pop(key,None)
if name=='codex':
    auth=Path(env.get('CODEX_AUTH_JSON_PATH',str(Path.home()/'.codex/auth.json'))).expanduser()
    assert auth.is_file(),'Set CODEX_AUTH_JSON_PATH to your existing private ChatGPT auth.json'
    env['CODEX_AUTH_JSON_PATH']=str(auth)
else:
    key=Path.home()/'.config/tb3/deepseek.key'
    assert key.is_file() and key.stat().st_mode&0o077==0,'Protect the existing DeepSeek key file with mode 600'
    audit=runtime/(args.id+'-request-audit');audit.mkdir()
    overlay={'services':{'main':{'depends_on':{'tb3-provider':{'condition':'service_healthy'}}},'tb3-provider':{
        'image':'python:3.13.7-slim-bookworm@sha256:adafcc17694d715c905b4c7bebd96907a1fd5cf183395f0ebc4d3428bd22d92d',
        'command':['python','-B','/relay.py'],'cpus':0.25,'mem_limit':'128m','read_only':True,
        'volumes':[{'type':'bind','source':str(repo/'scripts/provider_relay.py'),'target':'/relay.py','read_only':True},
            {'type':'bind','source':str(key),'target':'/run/secrets/deepseek.key','read_only':True},
            {'type':'bind','source':str(audit),'target':'/audit'}],
        'healthcheck':{'test':['CMD','python','-c',"import urllib.request; urllib.request.urlopen('http://127.0.0.1:8765/health',timeout=2).read()"],
            'interval':'2s','timeout':'3s','retries':10}}}}
    overlaypath=runtime/(args.id+'-provider.compose.json');overlaypath.write_text(json.dumps(overlay,indent=2)+'\n',encoding='utf-8')
    config['environment']['extra_docker_compose']=[str(overlaypath)]
    env.update(ANTHROPIC_API_KEY='tb3-local-provider-adapter',ANTHROPIC_BASE_URL='http://tb3-provider:8765/anthropic',
        ANTHROPIC_MODEL='deepseek-flash',ANTHROPIC_DEFAULT_OPUS_MODEL='deepseek-flash',
        ANTHROPIC_DEFAULT_SONNET_MODEL='deepseek-flash',ANTHROPIC_DEFAULT_HAIKU_MODEL='deepseek-flash',
        CLAUDE_CODE_SUBAGENT_MODEL='deepseek-flash',CLAUDE_CODE_EFFORT_LEVEL='max',CLAUDE_CODE_MAX_OUTPUT_TOKENS='128000')
configpath.write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
if args.print_config:print(json.dumps(config,indent=2));raise SystemExit(0)
raise SystemExit(subprocess.run(['harbor','run','--config',str(configpath),'--no-stream','--max-retries','0'],env=env,cwd=repo).returncode)
