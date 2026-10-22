#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-or-later
"""Prepare pinned official native agents without APT, npm, or a global install."""
from pathlib import Path
import argparse, base64, concurrent.futures, hashlib, io, json, os, subprocess, tarfile, urllib.request

p=argparse.ArgumentParser();p.add_argument('--directory',type=Path,default=Path.home()/'.cache/tb3-survival-agents');args=p.parse_args()
dest=args.directory.expanduser().resolve();dest.mkdir(parents=True,exist_ok=True)
specs=[('@openai/codex','0.163.0-linux-x64','codex'),('@anthropic-ai/claude-code-linux-x64','2.1.308','claude')]
def install(spec):
    package,version,name=spec
    url='https://registry.npmjs.org/'+package+'/'+version
    with urllib.request.urlopen(url,timeout=30) as response:meta=json.load(response)
    with urllib.request.urlopen(meta['dist']['tarball'],timeout=180) as response:data=response.read()
    integrity='sha512-'+base64.b64encode(hashlib.sha512(data).digest()).decode()
    assert integrity==meta['dist']['integrity'],'Registry integrity mismatch'
    directory=dest/(name+'-'+version)
    if directory.exists():raise RuntimeError(f'{directory} already exists; choose a new --directory rather than overwriting it')
    directory.mkdir()
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:
        for member in archive.getmembers():
            assert not member.issym() and not member.islnk()
            assert (directory/member.name).resolve().is_relative_to(directory)
        archive.extractall(directory,filter='data')
    native=next(v for v in directory.rglob(name) if v.is_file())
    native.chmod(0o755)
    executable=native
    if name=='codex':
        # Codex 0.163 also needs its sibling code-mode host and resources.
        # Keep the complete layout through a single read-only Harbor mount.
        header=r'''#!/bin/sh
exec python3 - "$0" "$@" <<'TB3_CODEX_EXTRACT'
from pathlib import Path
import os,sys,tarfile,io
base=Path('/tmp/tb3-official-codex-0.163.0')
vendor=base/'package/vendor/x86_64-unknown-linux-musl'
binary=vendor/'bin/codex'
if not (vendor/'bin/codex-code-mode-host').is_file():
    data=Path(sys.argv[1]).read_bytes().split(b'\nTB3_ARCHIVE_BEGIN\n',1)[1]
    base.mkdir(parents=True,exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as archive:archive.extractall(base,filter='data')
os.environ['PATH']=str(vendor/'codex-path')+':'+os.environ.get('PATH','')
os.execv(binary,[str(binary)]+sys.argv[2:])
TB3_CODEX_EXTRACT
TB3_ARCHIVE_BEGIN
'''
        executable=dest/'codex-complete-0.163.0'
        executable.write_bytes(header.encode()+data);executable.chmod(0o755)
    env=dict(os.environ,CODEX_HOME=str(dest/'version-check-home'))
    result=subprocess.run([str(executable),'--version'],capture_output=True,text=True,env=env,timeout=120)
    assert result.returncode==0,(result.stdout,result.stderr)
    return {'name':name,'version':version.split('-linux')[0],'executable':str(executable),
        'sha256':hashlib.sha256(executable.read_bytes()).hexdigest(),'native_sha256':hashlib.sha256(native.read_bytes()).hexdigest(),
        'registry_metadata':url,'tarball':meta['dist']['tarball'],'integrity':integrity,
        'version_command_exit_code':result.returncode,'version_output':result.stdout.strip()}
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:agents=list(executor.map(install,specs))
(dest/'agents.json').write_text(json.dumps(agents,indent=2)+'\n',encoding='utf-8')
for agent in agents:print(json.dumps({k:v for k,v in agent.items() if k!='executable'}))
