"""본 측정 보조 일꾼 (2026-10-07).

bt_b0_robust.py 와 같은 계획·같은 설정·같은 캐시를 쓴다. 다른 것은 순서뿐이다 —
계획을 뒤에서부터 훑고 샤드로 나눠 잡는다. 본 작업이 앞에서 오고 일꾼이 뒤에서 가서
가운데서 만난다. 이미 캐시에 있으면 건너뛰므로 겹쳐도 결과가 안 바뀐다.

쓰기는 임시파일 + rename 이라 원자적이다. 본 작업이 같은 키를 동시에 쓰더라도 내용이
결정적이라 최종 파일은 어느 쪽이 이기든 같다.

28코어 머신에서 네 프로세스까지 돌린 근거는 free -g available 24GB · load 1.3 이다.

실행: python3 b0_robust_worker.py <샤드번호> <샤드수>
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402
import bt_b0_robust as B  # noqa: E402


def ensure_atomic(sleeve: str, arm: str, ex, anchor: int, stress: bool) -> bool:
    k = B.key(sleeve, arm, ex, anchor, stress)
    path = os.path.join(B.EQDIR, f"{k}.csv")
    if B.load(k) is not None:
        return False
    os.environ["B0_ARM"] = "" if arm == "b0" else arm
    os.environ["SELECT_NO_ABSORB"] = "1" if arm == "fullpool" else "0"
    if ex:
        os.environ["B0_EXCLUDE"] = ",".join(ex)
    else:
        os.environ.pop("B0_EXCLUDE", None)
    try:
        if sleeve == "fu":
            eq = R.run_fut(B.FU_OVER, R.PH_FUT[anchor], stress)
        else:
            eq = R.run_spot(B.SP_OVER, R.PH_SPOT[anchor], stress)
    finally:
        os.environ["B0_ARM"] = ""
        os.environ["SELECT_NO_ABSORB"] = "0"
        os.environ.pop("B0_EXCLUDE", None)
    tmp = f"{path}.tmp{os.getpid()}"
    eq.to_csv(tmp, header=["eq"])
    os.replace(tmp, path)
    return True


def main() -> None:
    shard, nshards = int(sys.argv[1]), int(sys.argv[2])
    t0 = time.time()
    os.makedirs(B.EQDIR, exist_ok=True)
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    jobs = list(reversed(B.plan()))
    mine = [j for i, j in enumerate(jobs) if i % nshards == shard]
    print(f"샤드 {shard}/{nshards} · {len(mine)}조합 · 로드 {time.time()-t0:.0f}s",
          flush=True)
    ran = skip = 0
    for i, (sleeve, arm, ex) in enumerate(mine):
        for a in R.ANCHORS:
            for st in (False, True):
                if ensure_atomic(sleeve, arm, ex, a, st):
                    ran += 1
                else:
                    skip += 1
        if (i + 1) % 10 == 0 or i + 1 == len(mine):
            print(f"  {i+1}/{len(mine)}조합 · 새로 {ran} 건너뜀 {skip} "
                  f"· {(time.time()-t0)/60:.0f}분", flush=True)
    print(f"샤드 {shard} 완료 · 새로 {ran} 건너뜀 {skip} "
          f"· {(time.time()-t0)/60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
