**プロジェクト名**：LLMと音声合成によるゲームキャラクター対話システム「ヴェクターAI」

**概要** <br>大規模言語モデル（ELYZA Llama-3）と音声合成エンジン（VOICEVOX）を用いて，ドールズフロントラインに登場する「ヴェクター」を模した対話エージェントをPythonで開発しました．キャラクターの性格・話し方をプロンプトで定義し，ユーザーと自然な日本語で感情表現を交えた会話が可能です．

**使用技術** <br>Python<br> HuggingFace Transformers（ELYZA Llama-3モデル）<br> VOICEVOX（音声合成）

**ファイル説明** <br>vector.py：テキストのみの会話が可能．コマンドライン上で対話を行い，「終了」「exit」「quit」コマンドで会話を終了できます．<br>vector_voice.py：vector.pyに生成したテキストを読み上げる機能を追加したものになります．声を変更したい場合は，47行目の ```speaker = 4  # お好きな話者IDに変更してください（VOICEVOXで確認）```の数値を変更してださい．

**参考**<br> Llama-3：https://huggingface.co/elyza/Llama-3-ELYZA-JP-8B<br> VOICEVOX：https://github.com/cmqr/voicevox
