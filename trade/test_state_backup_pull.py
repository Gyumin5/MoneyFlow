#!/usr/bin/env python3
"""상태파일 백업 수신기와 점검기의 실패 경로를 고정한다 (2026-11-03).

왜 있나. 이 두 스크립트의 값은 "잘 받을 때" 가 아니라 "못 받을 때 티가 나나" 에 있다.
반쯤 받은 묶음이 완성본 이름을 갖거나, 깨진 JSON 이 통과하거나, 수신이 멈췄는데 점검이
정상이라고 말하면 백업이 있다는 믿음만 남고 실물이 없다. 그 자리를 하나씩 건다.

서버에는 붙지 않는다 — 원격 접근은 전부 가짜로 바꾸고 임시 디렉토리에서만 돈다.
"""
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from datetime import datetime, timedelta

OPS = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops')


def _load(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(OPS, name + '.py'))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


pull = _load('pull_state_backups')
health = _load('check_state_backup_health')

GOOD = {
    'signal_state.json': {'meta': {'updated_at': '2026-11-03'}, 'coin': {}, 'stock': {}},
    'trade_state.json': {'members': {}, 'schema_version': 'V24'},
    'kis_trade_state.json': {'snapshots': [], 'schema_version': 'V25'},
    'binance_state.json': {'strategies': {}, 'schema_version': 'V25'},
}


def make_cohort(path, overrides=None, drop=None):
    os.makedirs(path, exist_ok=True)
    for name, body in GOOD.items():
        if drop and name in drop:
            continue
        raw = (overrides or {}).get(name)
        with open(os.path.join(path, name), 'w', encoding='utf-8') as fh:
            if raw is None:
                json.dump(body, fh)
            else:
                fh.write(raw)
    with open(os.path.join(path, 'MANIFEST'), 'w', encoding='utf-8') as fh:
        fh.write('2026-11-03 09:10:02 KST\n')
    return path


def digests(path):
    return {n: pull.sha256_of(os.path.join(path, n)) for n in os.listdir(path)}


class TestVerify(unittest.TestCase):
    """verify_dir — 통과시켜서는 안 되는 것들."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def test_온전한_묶음은_통과한다(self):
        d = make_cohort(os.path.join(self.tmp, 'ok'))
        self.assertEqual(pull.verify_dir(d, digests(d)), [])

    def test_파일이_하나_빠지면_걸린다(self):
        d = make_cohort(os.path.join(self.tmp, 'drop'), drop={'binance_state.json'})
        self.assertIn('binance_state.json 없음', pull.verify_dir(d, digests(d)))

    def test_깨진_JSON_은_걸린다(self):
        d = make_cohort(os.path.join(self.tmp, 'broken'),
                        overrides={'trade_state.json': '{"members": '})
        self.assertTrue(any('JSON 깨짐' in p for p in pull.verify_dir(d, digests(d))))

    def test_빈_객체는_걸린다(self):
        d = make_cohort(os.path.join(self.tmp, 'empty'),
                        overrides={'signal_state.json': '{}'})
        self.assertTrue(any('빈 객체' in p for p in pull.verify_dir(d, digests(d))))

    def test_최상위가_목록이면_걸린다(self):
        d = make_cohort(os.path.join(self.tmp, 'list'),
                        overrides={'kis_trade_state.json': '[1,2,3]'})
        self.assertTrue(any('사전이 아님' in p for p in pull.verify_dir(d, digests(d))))

    def test_필수_키가_없으면_걸린다(self):
        d = make_cohort(os.path.join(self.tmp, 'nokey'),
                        overrides={'binance_state.json': '{"strategies": {}}'})
        self.assertTrue(any('필수 키 없음' in p for p in pull.verify_dir(d, digests(d))))

    def test_해시가_다르면_걸린다(self):
        d = make_cohort(os.path.join(self.tmp, 'hash'))
        exp = digests(d)
        exp['trade_state.json'] = '0' * 64
        self.assertIn('trade_state.json 해시 불일치', pull.verify_dir(d, exp))

    def test_파일_대신_심볼릭_링크면_없는_것으로_센다(self):
        d = make_cohort(os.path.join(self.tmp, 'link'), drop={'signal_state.json'})
        os.symlink('/etc/hostname', os.path.join(d, 'signal_state.json'))
        self.assertIn('signal_state.json 없음', pull.verify_dir(d, {}))

    def test_파일별로_다른_키를_본다(self):
        """넷에 같은 키 목록을 강제하면 엉뚱한 파일이 통과한다."""
        self.assertNotEqual(pull.REQUIRED['trade_state.json'],
                            pull.REQUIRED['kis_trade_state.json'])


class TestRemoteIndex(unittest.TestCase):
    """원격 목록 파싱 — 미완성 디렉토리를 완성본으로 읽으면 안 된다."""

    def test_날짜_이름만_받고_staging_은_버린다(self):
        out = ('aaa  ./2026-11-03/trade_state.json\n'
               'bbb  ./.staging_2026-11-04/trade_state.json\n'
               'ccc  ./2026-11-03/MANIFEST\n'
               'ddd  ./not-a-date/trade_state.json\n'
               'eee  ./2026-11-03/sub/deep.json\n')
        pull.run_ssh = lambda script: out
        idx = pull.remote_index()
        self.assertEqual(sorted(idx), ['2026-11-03'])
        self.assertEqual(sorted(idx['2026-11-03']), ['MANIFEST', 'trade_state.json'])

    def test_서버를_못_읽으면_예외가_올라온다(self):
        def boom(script):
            raise RuntimeError('ssh rc=255: Connection refused')
        pull.run_ssh = boom
        with self.assertRaises(RuntimeError):
            pull.remote_index()


class TestFetch(unittest.TestCase):
    """실패한 묶음에 최종 이름이 생기면 안 된다."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._dest = pull.DEST
        pull.DEST = self.tmp
        self.addCleanup(setattr, pull, 'DEST', self._dest)
        self.src = make_cohort(os.path.join(self.tmp, 'src'))
        self.exp = digests(self.src)

    def _fake_scp(self, broken=None, fail=False):
        src = self.src

        class R:
            returncode = 1 if fail else 0
            stderr = 'protocol error' if fail else ''

        def run(cmd, **kw):
            if fail:
                return R()
            name = os.path.basename(cmd[-2].split(':')[-1])
            dst = cmd[-1]
            if broken and name == broken:
                with open(dst, 'w', encoding='utf-8') as fh:
                    fh.write('{"members": ')      # 전송 중단을 흉내낸다
            else:
                shutil.copyfile(os.path.join(src, name), dst)
            return R()
        return run

    def test_정상_수신은_최종_이름을_갖는다(self):
        pull.subprocess.run = self._fake_scp()
        ok, memo = pull.fetch_cohort('2026-11-03', self.exp, dry=False)
        self.assertTrue(ok, memo)
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, '2026-11-03')))

    def test_전송이_끊기면_최종_이름이_안_생긴다(self):
        pull.subprocess.run = self._fake_scp(broken='trade_state.json')
        ok, _ = pull.fetch_cohort('2026-11-03', self.exp, dry=False)
        self.assertFalse(ok)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, '2026-11-03')))

    def test_scp_가_실패하면_임시_디렉토리도_안_남는다(self):
        pull.subprocess.run = self._fake_scp(fail=True)
        ok, _ = pull.fetch_cohort('2026-11-03', self.exp, dry=False)
        self.assertFalse(ok)
        self.assertEqual([d for d in os.listdir(self.tmp) if d.startswith('.incoming')], [])

    def test_받은_파일은_주인만_읽는다(self):
        pull.subprocess.run = self._fake_scp()
        pull.fetch_cohort('2026-11-03', self.exp, dry=False)
        final = os.path.join(self.tmp, '2026-11-03')
        self.assertEqual(os.stat(final).st_mode & 0o777, 0o700)
        for name in os.listdir(final):
            self.assertEqual(os.stat(os.path.join(final, name)).st_mode & 0o777, 0o600)


class TestCompareAndPrune(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        self._dest = pull.DEST
        pull.DEST = self.tmp
        self.addCleanup(setattr, pull, 'DEST', self._dest)

    def test_이미_있는_묶음이_다르면_사유를_돌려준다(self):
        d = make_cohort(os.path.join(self.tmp, '2026-11-03'))
        exp = digests(d)
        exp['trade_state.json'] = '0' * 64
        self.assertIn('trade_state.json 내용 다름',
                      pull.compare_existing('2026-11-03', exp))

    def test_같으면_빈_목록이다(self):
        d = make_cohort(os.path.join(self.tmp, '2026-11-03'))
        self.assertEqual(pull.compare_existing('2026-11-03', digests(d)), [])

    def _seed(self, count):
        base = datetime(2026, 11, 3)
        days = [(base - timedelta(days=i)).strftime('%Y-%m-%d') for i in range(count)]
        for day in days:
            os.makedirs(os.path.join(self.tmp, day))
        return sorted(days)

    def test_보관_기간을_넘긴_것만_지운다(self):
        self._seed(pull.KEEP_MIN + 1)      # 최근 일곱 개 보호와 겹치지 않게 넉넉히 둔다
        old = (datetime.now() - timedelta(days=pull.KEEP_DAYS + 5)).strftime('%Y-%m-%d')
        os.makedirs(os.path.join(self.tmp, old))
        removed = pull.prune({'cohorts': {}}, healthy=True)
        self.assertEqual(removed, [old])
        self.assertFalse(os.path.exists(os.path.join(self.tmp, old)))

    def test_오래됐어도_최근_일곱_개는_남긴다(self):
        base = datetime.now() - timedelta(days=pull.KEEP_DAYS + 30)
        for i in range(pull.KEEP_MIN):
            os.makedirs(os.path.join(self.tmp, (base + timedelta(days=i)).strftime('%Y-%m-%d')))
        self.assertEqual(pull.prune({'cohorts': {}}, healthy=True), [])

    def test_수신이_실패_중이면_아무것도_안_지운다(self):
        self._seed(3)
        old = (datetime.now() - timedelta(days=pull.KEEP_DAYS + 5)).strftime('%Y-%m-%d')
        os.makedirs(os.path.join(self.tmp, old))
        self.assertEqual(pull.prune({'cohorts': {}}, healthy=False), [])
        self.assertTrue(os.path.exists(os.path.join(self.tmp, old)))

    def test_복원_검증에_쓴_묶음은_기간이_지나도_남긴다(self):
        self._seed(8)
        old = (datetime.now() - timedelta(days=pull.KEEP_DAYS + 5)).strftime('%Y-%m-%d')
        os.makedirs(os.path.join(self.tmp, old))
        receipt = {'cohorts': {}, 'last_restore_check': {'cohort': old}}
        self.assertEqual(pull.prune(receipt, healthy=True), [])


class TestLock(unittest.TestCase):
    def test_같은_회차가_겹치면_3_으로_물러난다(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        for attr, val in (('DEST', tmp), ('RECEIPT', os.path.join(tmp, 'r.json')),
                          ('LOCK', os.path.join(tmp, '.lock'))):
            old = getattr(pull, attr)
            setattr(pull, attr, val)
            self.addCleanup(setattr, pull, attr, old)
        holder = open(pull.LOCK, 'w')
        self.addCleanup(holder.close)
        pull.fcntl.flock(holder, pull.fcntl.LOCK_EX | pull.fcntl.LOCK_NB)
        argv = sys.argv
        sys.argv = ['pull_state_backups.py']
        self.addCleanup(setattr, sys, 'argv', argv)
        self.assertEqual(pull.main(), 3)


class TestHealth(unittest.TestCase):
    """점검기 — 수신이 멈춘 것을 정상이라고 말하면 안 된다."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp, True)
        for attr, val in (('DEST', self.tmp), ('RECEIPT', os.path.join(self.tmp, 'receipt.json'))):
            old = getattr(health, attr)
            setattr(health, attr, val)
            self.addCleanup(setattr, health, attr, old)

    def _write(self, day, run_ts, error=''):
        make_cohort(os.path.join(self.tmp, day))
        with open(health.RECEIPT, 'w', encoding='utf-8') as fh:
            json.dump({'last_run': run_ts.strftime('%Y-%m-%d %H:%M:%S'),
                       'last_error': error}, fh)

    def test_방금_받았으면_정상이다(self):
        now = datetime.now()
        self._write(now.strftime('%Y-%m-%d'), now)
        self.assertEqual(health.diagnose(now), [])

    def test_하나도_없으면_경보다(self):
        self.assertTrue(health.diagnose(datetime.now()))

    def test_오후_늦게_오늘_묶음이_없으면_경보다(self):
        now = datetime(2026, 11, 3, 16, 0)
        self._write('2026-11-02', now)
        self.assertTrue(any('오늘' in b for b in health.diagnose(now)))

    def test_같은_상태라도_이른_시각에는_안_울린다(self):
        now = datetime(2026, 11, 3, 10, 0)
        self._write('2026-11-02', now)
        self.assertFalse(any('오늘' in b for b in health.diagnose(now)))

    def test_최신이_너무_오래되면_경보다(self):
        now = datetime(2026, 11, 3, 12, 0)
        self._write('2026-10-31', now)
        self.assertTrue(any('시간 지났다' in b for b in health.diagnose(now)))

    def test_수신기가_멈추면_경보다(self):
        now = datetime.now()
        self._write(now.strftime('%Y-%m-%d'), now - timedelta(hours=5))
        self.assertTrue(any('안 돌았다' in b for b in health.diagnose(now)))

    def test_지난_회차_실패가_남아_있으면_경보다(self):
        now = datetime.now()
        self._write(now.strftime('%Y-%m-%d'), now, error='2026-11-03: 해시 불일치')
        self.assertTrue(any('실패가 남아' in b for b in health.diagnose(now)))

    def test_필수_파일이_빠지면_경보다(self):
        now = datetime.now()
        day = now.strftime('%Y-%m-%d')
        make_cohort(os.path.join(self.tmp, day), drop={'trade_state.json'})
        with open(health.RECEIPT, 'w', encoding='utf-8') as fh:
            json.dump({'last_run': now.strftime('%Y-%m-%d %H:%M:%S'), 'last_error': ''}, fh)
        self.assertTrue(any('빠진 파일' in b for b in health.diagnose(now)))

    def test_기록을_못_읽으면_경보다(self):
        now = datetime.now()
        make_cohort(os.path.join(self.tmp, now.strftime('%Y-%m-%d')))
        self.assertTrue(any('receipt' in b for b in health.diagnose(now)))


if __name__ == '__main__':
    unittest.main(verbosity=2)
