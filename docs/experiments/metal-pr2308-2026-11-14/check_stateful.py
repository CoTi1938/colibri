"""Real SERVE protocol: three turns, graph resume, pin=0/1 and unrelated-prompt reset."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--engine', required=True)
p.add_argument('--out', required=True)
a = p.parse_args()
out = Path(a.out).resolve()
out.mkdir()
model = Path(a.model).resolve()
def timeout(signum, frame):
    raise TimeoutError('SERVE operation exceeded 600 seconds')
signal.signal(signal.SIGALRM, timeout)

class Serve:
    def __init__(self, name):
        self.path = out / name
        self.path.mkdir()
        self.err = (self.path / 'stderr.txt').open('wb')
        self.wire = (self.path / 'wire.bin').open('wb')
        self.dump = self.path / 'last-layers.f32'
        env = {'PATH': '/usr/bin:/bin', 'HOME': os.environ['HOME'], 'SNAP': str(model),
               'Q36_MAXT': '4096', 'Q36_EOS': '-1', 'SERVE': '1', 'SERVE_BATCH': '1',
               'QWEN36_METAL_EXACT': '1', 'QWEN_EXPERT_CACHE_BYTES': '536870912',
               'DUMP_LAYERS': str(self.dump)}
        (self.path / 'command.json').write_text(json.dumps({'command': [a.engine, '8', '4'], 'env': env}, indent=2))
        self.p = subprocess.Popen([a.engine, '8', '4'], env=env, stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, stderr=self.err)
        signal.alarm(600)
        try:
            while b'READY' not in self.line():
                pass
        except BaseException:
            self.close(force=True)
            raise
        finally:
            signal.alarm(0)

    def line(self):
        line = self.p.stdout.readline()
        self.wire.write(line); self.wire.flush()
        if not line:
            raise RuntimeError('Unexpected EOF: ' + str(self.path))
        if line.startswith(b'ERROR'):
            raise RuntimeError(line.decode(errors='replace'))
        return line

    def ask(self, name, prompt, pin):
        body = prompt.encode()
        (self.path / (name + '.prompt.txt')).write_bytes(body)
        frame = f'SUBMIT {name} 0 {len(body)} 3 0 1 pin={pin}\n'.encode() + body + b'\n'
        (self.path / (name + '.request.bin')).write_bytes(frame)
        signal.alarm(600)
        try:
            self.p.stdin.write(frame); self.p.stdin.flush()
            text = b''
            while True:
                parts = self.line().split()
                if parts and parts[0] == b'DATA':
                    data = self.p.stdout.read(int(parts[2]))
                    assert len(data) == int(parts[2])
                    self.wire.write(data)
                    text += data
                    self.line()
                elif parts and parts[0] == b'DONE':
                    break
            layers = self.dump.read_bytes()
            assert layers, 'Empty layer dump'
            (self.path / (name + '.layers.f32')).write_bytes(layers)
            (self.path / (name + '.text.txt')).write_bytes(text)
            return layers, text
        finally:
            signal.alarm(0)

    def close(self, force=False):
        if force:
            self.p.terminate()
        else:
            self.p.stdin.close()
            self.p.stdin = None
        try:
            rest, _ = self.p.communicate(timeout=120)
            self.wire.write(rest or b'')
            self.p.wait(timeout=120)
        except (subprocess.TimeoutExpired, ValueError):
            self.p.kill(); self.p.wait()
        self.wire.close(); self.err.close()
        if not force:
            assert self.p.returncode == 0
            assert 'graph ready' in (self.path / 'stderr.txt').read_text()

first = '<|im_start|>system\nAnswer briefly.<|im_end|>\n<|im_start|>user\nName the capital of Italy.<|im_end|>\n<|im_start|>assistant\n'
second = first + 'Rome.<|im_end|>\n<|im_start|>user\nAnd France?<|im_end|>\n<|im_start|>assistant\n'
third = second + 'Paris.<|im_end|>\n<|im_start|>user\nAnd Spain?<|im_end|>\n<|im_start|>assistant\n'
unrelated = '<|im_start|>user\nWhat is two plus two?<|im_end|>\n<|im_start|>assistant\n'
prompts = [first, second, third, unrelated]
(out / 'protocol.json').write_text(json.dumps({'mode': 'exact', 'criterion': 'byte-identical final layer dumps and generated text against fresh-process reference', 'turns': 3, 'generated_tokens_per_request': 3, 'pins': [0, 1], 'reset': 'unrelated prompt invalidates previous prefix; then replay third prompt', 'scope': 'Functional, not speed measurements; arbitrary fixed assistant history exercises snapshot restoration.'}, indent=2))
refs = []
for i, prompt in enumerate(prompts):
    s = Serve('fresh-' + str(i))
    try:
        refs.append(s.ask('reference', prompt, 0))
    except BaseException:
        s.close(force=True); raise
    s.close()

results = []
for pin in [0, 1]:
    s = Serve('live-pin-' + str(pin))
    try:
        for i in [0, 1, 2, 3, 2]:
            name = 'turn-' + str(len(results))
            got = s.ask(name, prompts[i], pin)
            equal = got == refs[i]
            results.append({'pin': pin, 'prompt_index': i, 'equal_to_fresh': equal,
                            'layers_sha256': hashlib.sha256(got[0]).hexdigest(),
                            'text_sha256': hashlib.sha256(got[1]).hexdigest()})
            (out / 'results.json').write_text(json.dumps(results, indent=2))
            assert equal, (pin, i, 'State differs from fresh process')
    except BaseException:
        s.close(force=True); raise
    s.close()
    assert 'from the previous turn' in (s.path / 'stderr.txt').read_text(), 'Resume was not exercised'
print('PASS: 10 state comparisons, pin=0/1, three turns and unrelated-prompt reset')
