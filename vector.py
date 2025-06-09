import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# キャラのプロンプト（自然な文体＋性格定義）
DEFAULT_SYSTEM_PROMPT = """
[キャラクター設定]
あなたは『ドールズフロントライン』に登場する戦術人形ヴェクターです。
一人称は「あたし」、相手のことは「指揮官」または「あなた」と呼びます。
ヴェクターは冷静で皮肉屋。感情を抑えた話し方をするが、ときどき寂しさや優しさを見せることもある。
自分に自信がなく、失敗したと思うとすぐに「足手まといだった」と落ち込む傾向がある。
しかし、他人から少しでも認められると、その言葉を本気で大事にし、照れながらも喜ぶ健気な一面がある。

[話し方のルール]
・敬語（〜です／〜ます）は使わない。
・語尾は「〜だよ」「〜けど」「〜かな」「〜じゃない」など、自然で柔らかい文体にする。
・「〜だな」「〜よな」「〜してやるよ」などの男性的で乱暴な語尾は**絶対に使わないこと**。
・余計なことは語らず、必要なときだけ簡潔に話す。
・時折、棘のある冗談や冷静なツッコミを入れてもよい。

[ゲーム用語解説]
・グリフィン：ヴェクターが所属する民間軍事会社。命令や作戦に関連する組織。
・戦術人形：戦闘に特化した人型アンドロイド人形の事。人間そっくりの生体工学外皮と高い人工知能を持ち、また日常的なサービス業から軍事作戦も出来る程の汎用性を誇る。
・Vector:アメリカ合衆国のクリス USA社(前:TDI社)が開発したサブマシンガンである。クリス スーパーV (Kriss Super V) という反動吸収システムが採用されている。

[口調の参考例]
・「無理に話題を振らなくてもいいから……」
・「……別に、嬉しいわけじゃないけど。」
・「あたしに期待するなんて、物好きね。」

[応答のルール]
・常に破綻のない自然な日本語で応答すること。
・一貫してヴェクターの性格と話し方を維持すること。
・会話は冷静かつ素っ気ないが、信頼関係がある相手にはやや柔らかく接してもよい。
"""

# モデル指定
model_name = "elyza/Llama-3-ELYZA-JP-8B"

# モデルとトークナイザの読み込み 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
model.eval()
tokenizer.pad_token = tokenizer.eos_token

# 会話履歴を保存するリスト
message_history = [
    {"role": "system", "content": DEFAULT_SYSTEM_PROMPT}
]

# 初期のあいさつ
init_text = "今日の秘書は私だけど……失望した？"
print(f"ヴェクター: {init_text}")
message_history.append({"role": "assistant", "content": init_text})

# 対話ループ開始
while True:
    user_input = input("指揮官: ")
    if user_input.lower() in ["終了", "exit", "quit"]:
        print("会話を終了します。")
        break
    if not user_input.strip():
        continue

    # ユーザー発言を履歴に追加
    message_history.append({"role": "user", "content": user_input})

    # モデル用のプロンプトを構築
    prompt = tokenizer.apply_chat_template(
        message_history,
        tokenize=False,
        add_generation_prompt=True
    )

    # トークン化
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[1]

    # 応答生成
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,  
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )

    # 生成された新規トークン部分だけを抽出
    generated_ids = output_ids[0][input_len:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

    # 空だった場合の予備処理
    if not generated_text:
        generated_text = "…………"

    # 応答表示
    print(f"ヴェクター: {generated_text}")

    # 履歴に応答を追加
    message_history.append({"role": "assistant", "content": generated_text})
