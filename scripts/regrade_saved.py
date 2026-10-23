"""Regrade a published saved implementation without calling any model provider.

Run on Linux with Docker. This is an offline artifact check, not a model trial.
The supplied verifier still owns grading and launches candidate code as UID 10001.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import json
import subprocess
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
NAMES = ('codex-standard-1-replacement', 'codex-standard-2', 'codex-standard-3',
         'deepseek-standard-1', 'deepseek-standard-2', 'deepseek-standard-3', 'alternative')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trial', choices=NAMES)
    parser.add_argument('--output', type=Path, required=True, help='New directory for private execution logs and CTRF')
    parser.add_argument('--image', help='Already built verifier image ID or tag; otherwise build from this checkout')
    args = parser.parse_args()
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output/'verifier'; logs.mkdir(mode=0o700)
    source = (ROOT/'reports/reference-implementations/alternative' if args.trial=='alternative'
              else ROOT/'reports/trials'/args.trial/'submission')
    assert (source/'sksurv/metrics.py').is_file()
    record = {'kind':'offline artifact regrade; not a model trial', 'new_model_inference':False,
              'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'implementation':args.trial,
              'source_sha256':hashlib.sha256((source/'sksurv/metrics.py').read_bytes()).hexdigest(),
              'commands':[]}

    def run(label, argv, timeout):
        started = time.monotonic()
        with (output/(label+'.stdout.log')).open('wb') as stdout, (output/(label+'.stderr.log')).open('wb') as stderr:
            try:
                result = subprocess.run(argv, stdout=stdout, stderr=stderr, timeout=timeout)
                code = result.returncode
            except subprocess.TimeoutExpired:
                code = 124
                if label == 'verify':
                    subprocess.run(['docker','rm','--force',record['container_name']],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=30)
        record['commands'].append({'command':argv,'native_exit_code':code,'elapsed_sec':time.monotonic()-started})
        (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
        if code:
            raise RuntimeError(f'{label} exited {code}; inspect logs. No valid model-failure claim follows.')

    image = args.image or 'tb3-survival-concordance-offline-verifier:local'
    if not args.image:
        run('build',['docker','build','--progress=plain','-t',image,str(ROOT/'tasks/survival-concordance/tests')],900)
    record['image_id'] = subprocess.check_output(['docker','image','inspect','--format','{{.Id}}',image],text=True).strip()
    record['container_name'] = 'tb3-offline-'+uuid.uuid4().hex[:12]
    run('verify',['docker','run','--rm','--name',record['container_name'],'--network','none','--cpus','1','--memory','2g',
                  '--mount',f'type=bind,source={source},target=/app/work,readonly',
                  '--mount',f'type=bind,source={logs},target=/logs/verifier',
                  record['image_id'],'bash','/tests/test.sh'],1250)
    ctrf = json.loads((logs/'ctrf.json').read_text())
    summary = ctrf['results']['summary']; tests = ctrf['results']['tests']
    assert len(tests)==len({t['name'] for t in tests})==summary['tests']
    for status in ('passed','failed','skipped','pending','other'):
        assert summary[status]==sum(t['status']==status for t in tests)
    reward = float((logs/'reward.txt').read_text())
    assert reward in (0.,1.) and reward==int(summary['failed']==0)
    record.update(reward=reward, ctrf=summary,
                  finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (output/'result.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':
    main()
