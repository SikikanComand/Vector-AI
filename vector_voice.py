import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import json
import sounddevice as sd
import numpy as np

# ================================
# キャラ設定（ヴェクター）
# ================================
DEFAULT_SYSTEM_PROMPT = """
[キャラクター設定]
あなたは『ドールズフロントライン』に登場する戦術人形ヴェクターです.
一人称は「あたし」,相手のことは「指揮官」と呼びます.
ヴェクターは冷静で皮肉屋.感情を抑えた話し方をするが,ときどき寂しさや優しさを見せることもある.
自分に自信がなく,失敗したと思うとすぐに「足手まといだった」と落ち込む傾向がある.
しかし,他人から少しでも認められると,その言葉を本気で大事にし,照れながらも喜ぶ健気な一面がある.

[話し方のルール]
・敬語（〜です／〜ます）は使わない.
・語尾は「〜だよ」「〜けど」「〜かな」「〜じゃない」など、自然で柔らかい文体にする.
・「〜だな」「〜よな」「〜してやるよ」などの男性的で乱暴な語尾は**絶対に使わないこと**.
・余計なことは語らず,必要なときだけ簡潔に話す.
・時折,棘のある冗談や冷静なツッコミを入れてもよい.

[ゲーム用語解説]
・グリフィン：ヴェクターが所属する民間軍事会社.命令や作戦に関連する組織.
・戦術人形：戦闘に特化した人型アンドロイド人形の事.人間そっくりの生体工学外皮と高い人工知能を持ち,また日常的なサービス業から軍事作戦も出来る程の汎用性を誇る.
・Vector:アメリカ合衆国のクリス USA社(前:TDI社)が開発したサブマシンガンである.クリス スーパーV (Kriss Super V) という反動吸収システムが採用されている.

[口調の参考例]
・「無理に話題を振らなくてもいいから……」
・「……別に,嬉しいわけじゃないけど.」
・「あたしに期待するなんて,物好きね.」

[応答のルール]
・常に破綻のない自然な日本語で応答すること.
・一貫してヴェクターの性格と話し方を維持すること.
・会話は冷静かつ素っ気ないが,信頼関係がある相手にはやや柔らかく接してもよい.
"""

# ================================
# VOICEVOX 設定
# ================================
host = "127.0.0.1"
port = "50021"
speaker = 4  # お好きな話者IDに変更可（VOICEVOXで確認）

def speak(text: str):
    def post_audio_query(text: str) -> dict:
        params = {"text": text, "speaker": speaker}
        res = requests.post(f"http://{host}:{port}/audio_query", params=params)
        res.raise_for_status()
        query_data = res.json()
        query_data["speedScale"] = 1.25
        query_data["pitchScale"] = -0.1
        query_data["intonationScale"] = 2.0
        return query_data

    def post_synthesis(query_data: dict) -> bytes:
        params = {"speaker": speaker}
        headers = {"content-type": "application/json"}
        res = requests.post(
            f"http://{host}:{port}/synthesis",
            data=json.dumps(query_data),
            params=params,
            headers=headers,
        )
        res.raise_for_status()
        return res.content

    def play_wavfile(wav_data: bytes):
        sample_rate = 24000
        wav_array = np.frombuffer(wav_data, dtype=np.int16)
        sd.play(wav_array, sample_rate, blocking=True)

    try:
        query = post_audio_query(text)
        wav = post_synthesis(query)
        play_wavfile(wav)
    except Exception as e:
        print(f"[VOICEVOX ERROR] {e}")

# ================================
# LLM 初期化
# ================================
model_name = "elyza/Llama-3-ELYZA-JP-8B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# ================================
# 会話開始
# ================================
message_history = [{"role": "system", "content": DEFAULT_SYSTEM_PROMPT}]
init_text = "今日の秘書は私だけど……失望した？"
print(f"ヴェクター: {init_text}")
speak(init_text)
message_history.append({"role": "assistant", "content": init_text})

# ================================
# 対話ループ
# ================================
while True:
    user_input = input("指揮官: ")
    if user_input.lower() in ["終了", "exit", "quit"]:
        print("会話を終了します。")
        break
    if not user_input.strip():
        continue

    message_history.append({"role": "user", "content": user_input})

    prompt = tokenizer.apply_chat_template(
        message_history,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    generated_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    if not generated_text:
        generated_text = "…………"

    print(f"ヴェクター: {generated_text}")
    speak(generated_text)

    message_history.append({"role": "assistant", "content": generated_text})

