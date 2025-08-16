import os, lmdb, sys, textwrap

cache_filename = os.path.join(
    r"C:\Users\DGIST\Downloads\visualnav-transformer-main"
    r"\visualnav-transformer-main\train\data_splits\recon\train",
    "dataset_recon.lmdb",
)

print("📂 캐시 경로:", cache_filename)
print("💾 파일 존재? ", os.path.exists(cache_filename))

if os.path.exists(cache_filename):
    try:
        env = lmdb.open(cache_filename, readonly=True, lock=False)
        with env.begin() as txn:
            stat = env.stat()
            print("✅ entries:", stat["entries"])
            cur = txn.cursor()
            first = next(iter(cur), None)
            print("✅ sample key:", None if first is None else first.decode()[:120])
    except Exception as e:
        print("❌ LMDB 열기 실패:", e)
else:
    print(textwrap.dedent("""
        ❌ LMDB 파일이 없습니다.
        →   (1) _build_caches() 안의 if 블록이 실행되지 않았거나
        →   (2) 다른 폴더에 캐시가 생성됐을 수 있습니다.
    """))
