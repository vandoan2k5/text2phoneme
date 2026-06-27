# text2phoneme

Seq2seq encoder-decoder Transformer cho bài toán text-to-phoneme, dùng `piper_phonemize` để sinh nhãn phoneme và hỗ trợ dữ liệu interleaved Việt/Anh theo thẻ `<en>...</en>`.

## Ý tưởng chính

- Input text dùng tokenizer byte-level BPE trên chuỗi đã bỏ `<en>` và `</en>`.
- Target phoneme dùng vocabulary có sẵn trong `resource/phoneme_*.txt`.
- Decoder dùng hai namespace phoneme tách biệt cho `vi` và `en`; cùng một ký hiệu phoneme nhưng id target khác nhau theo ngôn ngữ.
- Với câu interleaved, pipeline sẽ:
  - Tách segment theo thẻ `<en>...</en>`.
  - Gọi phonemizer `vi` cho phần ngoài thẻ.
  - Gọi phonemizer `en` cho phần trong thẻ.
  - Ghép chuỗi phoneme theo đúng thứ tự ban đầu.
  - Trước khi đưa vào model, bỏ toàn bộ thẻ ngôn ngữ khỏi input text.
- Mô hình là encoder-decoder theo phong cách GPT-2:
  - learned token embedding + learned positional embedding,
  - pre-layernorm residual blocks,
  - self-attention/MLP kiểu GPT-2,
  - decoder có thêm cross-attention để làm seq2seq.

## File chính

- [dataset.py](/kaggle/text2phoneme/dataset.py)
- [model/net.py](/kaggle/text2phoneme/model/net.py)
- [train.py](/kaggle/text2phoneme/train.py)
- [inference.py](/kaggle/text2phoneme/inference.py)
- [model/model.yaml](/kaggle/text2phoneme/model/model.yaml)
- [train.yaml](/kaggle/text2phoneme/train.yaml)
- [smoke.yaml](/kaggle/text2phoneme/smoke.yaml)

## Cấu hình

- `model/model.yaml`: chỉ chứa kiến trúc model.
- `train.yaml`: cấu hình huấn luyện chính, tham chiếu `model_config: model/model.yaml`.
- `smoke.yaml`: cấu hình smoke test, có thể override một phần kiến trúc nhỏ hơn.
- `data.dataset_sample_limits`: giới hạn số mẫu riêng cho từng dataset con.

## Chạy train

```bash
python train.py --config train.yaml
```

Checkpoint và tokenizer text sẽ được ghi vào `artifacts/`.

## Chạy inference

```bash
python inference.py --checkpoint artifacts/best.pt --text "Là một <en>office worker</en>."
```

## Gợi ý cho 2x T4

- Smoke test: giữ cấu hình hiện tại trong `model/model.yaml`.
- Train lớn hơn:
  - tăng `limit_per_dataset` hoặc bỏ giới hạn,
  - tăng `epochs`,
  - nếu thiếu VRAM, giảm `batch_size` rồi tăng `num_workers` lên `4`,
  - nếu cần throughput tốt hơn, có thể tiền xử lý và cache phoneme trước khi train dài.
