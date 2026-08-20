# Synthetic Learning Events

`learning_events.csv.gz` は外部公開デモ専用の完全な合成データです。元データのレコード、ユーザーID、教材ID、日時、個人属性は複製していません。

生成方法は [`scripts/generate_public_dummy_data.py`](../scripts/generate_public_dummy_data.py) に固定し、`metadata.json` に件数とシードを記録しています。

主な列:

- `userid`: `DEMO-*` で始まる架空ID
- `course_id`: 架空コースID
- `week`: 1〜8週目
- `operationname`: 合成した教材操作
- `eventtime`: 架空日時
- `grade`: 合成した成績
- `synthetic_profile`: 生成時の行動プロファイル
- `is_injected_anomaly`, `anomaly_scenario`: 生成時に挿入した外れ行動の検証用ラベル

公開アプリの異常検知は、検証用ラベルを特徴量として使用しません。
