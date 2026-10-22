"""Local provider adapter: forward unchanged API bodies, audit allowlisted metadata.

The real key is held outside the agent container. No messages, headers, key, or
response content are logged. TLS validation remains enabled to official DeepSeek.
"""
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json, threading, time, urllib.request, urllib.error

# Controller admission stop: no key is read and no provider request is sent.
if Path('/audit/CANCELED.json').exists():
    raise SystemExit('Trial canceled by controller before provider initialization; see local cancellation record.')
KEY=Path('/run/secrets/deepseek.key').read_text().strip()
LOCK=threading.Lock()
DESTINATION='https://api.deepseek.com'
def audit(row):
    row['utc']=datetime.now(timezone.utc).isoformat()
    with LOCK,Path('/audit/requests.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')

class Handler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.0'
    def log_message(self,*args):pass
    def do_GET(self):
        if self.path=='/health':
            self.send_response(200);self.end_headers();self.wfile.write(b'OK');return
        self.send_error(404)
    def do_POST(self):
        started=time.monotonic()
        path=self.path.split('?')[0]
        if path not in ('/anthropic/v1/messages','/anthropic/v1/messages/count_tokens',
                        '/anthropic/messages','/anthropic/messages/count_tokens'):
            self.send_error(404);return
        length=int(self.headers.get('Content-Length','0'))
        if not 0<length<32*1024*1024:self.send_error(413);return
        body=self.rfile.read(length)
        try:request_body=json.loads(body)
        except Exception:self.send_error(400);return
        model=request_body.get('model')
        row={'path':path,'destination':DESTINATION+path,'model':model,
             'effort':(request_body.get('output_config') or {}).get('effort'),
             'thinking':(request_body.get('thinking') or {}).get('type'),
             'max_tokens':request_body.get('max_tokens'),'stream':request_body.get('stream',False)}
        # Enforce the user's explicit model spending authorization only.
        if model not in ('deepseek-flash','deepseek-flash[1m]'):
            row.update(status=400,local_error='model outside authorized flash configuration');audit(row)
            self.send_error(400,'Only the authorized DeepSeek Flash model is configured');return
        headers={'Content-Type':'application/json','x-api-key':KEY,
                 'anthropic-version':self.headers.get('anthropic-version','2023-06-01')}
        if self.headers.get('anthropic-beta'):headers['anthropic-beta']=self.headers['anthropic-beta']
        req=urllib.request.Request(DESTINATION+self.path,data=body,headers=headers,method='POST')
        try:
            try:response=urllib.request.urlopen(req,timeout=600)
            except urllib.error.HTTPError as error:response=error
            row['status']=response.status
            self.send_response(response.status)
            for name in ('Content-Type','Content-Length','Retry-After','request-id'):
                if response.headers.get(name):self.send_header(name,response.headers[name])
            self.send_header('Connection','close');self.end_headers()
            with response:
                while True:
                    chunk=response.read1(16384)
                    if not chunk:break
                    self.wfile.write(chunk);self.wfile.flush()
        except Exception as error:
            # Exception types only: never serialize requests/headers or key.
            row['transport_error']=type(error).__name__
        finally:
            row['elapsed_sec']=time.monotonic()-started;audit(row)

ThreadingHTTPServer(('0.0.0.0',8765),Handler).serve_forever()
