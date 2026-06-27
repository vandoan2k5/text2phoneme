from datasets import load_dataset
dataset = load_dataset("luvox-ai/interleaved_vi_en", "usa_persona")
dataset.save_to_disk("interleaved_vi_en/english_subset")
print("Đã lưu xong!")