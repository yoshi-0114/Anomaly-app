# 公開データ

`learning_events.csv.gz` は公開デモで読み込むイベントログです。

## 列の意味

- `userid`: 学生識別子
- `course_id`: コース識別子
- `week`: 対象週
- `contentsid`: 教材識別子
- `operationname`: 教材上の操作種別
- `pageno`: 操作対象のページ番号
- `marker`: マーカーの種別（空欄の場合あり）
- `memo_length`: メモの文字数
- `devicecode`: 利用端末の区分
- `eventtime`: 操作時刻
- `grade`: 評価区分
- `synthetic_profile`: 行動傾向の区分
- `is_injected_anomaly`: 検証用フラグ（0または1）
- `anomaly_scenario`: 検証用の区分（空欄の場合あり）
