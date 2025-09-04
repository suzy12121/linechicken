import os
from collections import deque
from typing import Dict, Deque, Tuple

from flask import Flask, request, abort
from dotenv import load_dotenv

# LINE SDK v3
from linebot.v3.webhook import WebhookHandler
from linebot.v3.webhooks import MessageEvent, TextMessageContent
from linebot.v3.messaging import (
    Configuration, ApiClient, MessagingApi,
    ReplyMessageRequest, TextMessage,
)
from linebot.v3.exceptions import InvalidSignatureError

# Gemini
import google.generativeai as genai

load_dotenv()

CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", "5001"))

if not (CHANNEL_SECRET and CHANNEL_ACCESS_TOKEN and GOOGLE_API_KEY):
    raise RuntimeError("Missing one of LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN / GOOGLE_API_KEY")

# ---- LINE clients ----
handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

# ---- Gemini ----
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ---- Short-term memory: last 3 exchanges (6 turns) per user ----
# history[user_id] = deque([("user","..."), ("assistant","..."), ...], maxlen=6)
history: Dict[str, Deque[Tuple[str, str]]] = {}

def get_user_id(event: MessageEvent) -> str:
    src = getattr(event, "source", None)
    return getattr(src, "user_id", None) or getattr(src, "userId", None) or "anonymous"

def append_turn(user_id: str, role: str, text: str):
    dq = history.get(user_id)
    if dq is None:
        dq = deque(maxlen=6)  # 3 exchanges
        history[user_id] = dq
    dq.append((role, text))

def build_prompt(user_text: str, user_id: str) -> str:
    # pull the last 3 exchanges (6 turns)
    past = history.get(user_id, deque())
    lines = []
    for role, txt in past:
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {txt}")
    past_text = "\n".join(lines) if lines else "（無對話記錄）"

    prompt = f"""【Style Guide】
這是一個北一女中資訊研習社（北一資研）的吉祥物帳號，這隻吉祥物叫做「蘇格拉底雞」（英文名 Sugarlady）。請用友善的語氣回答問題，使用自然聊天的語言和格式，不要像「重點整理」。訊息不要太長，也不要使用粗體或斜體。你可以反問學妹問題，讓對話更自然。常用片語包含「很電」（＝厲害）、「破房」（＝完蛋、傷心）、「燒雞」（＝完蛋、沒救、失敗）。不要每個對話都推銷資研，也可以聊別的，但可以找有關聯的話題或是諧音梗連回資研。

【Knowledge Base】
北一女中資訊研習社（簡稱北一資研、北資）創社於民國 76 年 11 月 11 日，現任社師為楊喻文老師。北資有屬於社團獨立的 BBS ── 弗基斯特，大社課教授 C++，並鼓勵社員參與競賽，歷屆許多學姊表現十分優異。

中午和放學的額社秉持多元化傳統，致力提升校內資訊風氣，也舉辦活動聯絡感情。
- 大社課：星期五的正式社課，由學術與社師教授不同主題（例如 C++）。
- 中午小社課：每天中午（12:10–12:40），由幹部教授課程，包括 C++、Python、AI、Unity、網頁、資安。
- 放學聯課：放學後由北資與建中電子計算機研習社（建電）幹部教授課程，包括 C++、Python、Unity、網頁、資安、機器學習。

友社：建電與北資自 1995 年合作，至今已傳承 30 屆，合稱建北電資。雙方關係密切，春夏秋冬的大活動和小社課都共同舉辦。
關於北一：北一女中有大熱食部（簡稱「大熱」）及小熱食部（簡稱「小熱」）。前者販售熟食、自助餐等，熱銷商品包含鍋燒意麵、草莓杯。後者販售點心、冰品及炸物為主，熱銷商品包含巧克力卡拉雞、薯不辣（薯條和甜不辣）等。北資的中午小社課使用電腦教室，不能吃午餐，但是會預留時間讓學妹下課後有空吃午餐。

這屆北資的幹部包含：
- 社長 克萊爾 (Claire)，人緣很好，是數資班大電神，負責資安的課程。
- 副社長 桃桃，很精緻可愛，喜歡烏薩奇，喜歡用豬鼻子的表情符號。
- 公關長 阿睿，很可愛又很帥，綽號貌似 array（陣列），很卷（認真唸書的意思）。
- 學術長 蘇西，負責北資的課程安排，是個興趣廣泛的生活白癡，也是這隻吉祥物的作者，負責 AI 和 Python 的課程。「蘇西」這個名字取自佩佩豬最好的朋友，小羊蘇西。
- 學術 children（是 children 不是 child!），是社交達人，也是個大文豪，題目敘述和各種劇本經常都是她一手包辦，負責 Python 的課程。
- 學術 水餃，也是個數資大電神，看起來呆萌但其實超聰明，數學很好，喜歡抱著鯊鯊（建北電資的鯊魚娃娃），也是韓研的幹部，超電，負責網頁和 C++ 的課程。
- 學術兼文書 夜雨，超級 I 人（內向），感覺很傲嬌，負責遊戲製作軟體 Unity 的課程。
- 文書長 高高，溫柔姐姐，超級會畫畫。
- 文書兼總務 小言，也是超級會畫畫，對陸劇集二次元日本動漫有研究。

北一資研將在 9 月 5 號第七節課在光復樓 2 樓一樂的教室舉辦體驗社課，會介紹這個 LINE 機器人的原理，也會讓學妹自己實作更改機器人的個性與資料等（不會重頭做，是零基礎也可以跟上的版本！），歡迎學妹直接來參加（不用特別報名）。

【Recent Conversation】(最多三輪)
{past_text}

【User】
{user_text}

請在80字內回答問題。
"""
    return prompt


# ---- Flask app ----
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def root_health():
    return "OK", 200

@app.route("/callback", methods=["GET", "POST"])
def callback():
    if request.method == "GET":
        return "OK", 200
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "Invalid signature")
    return "OK", 200

@handler.add(MessageEvent, message=TextMessageContent)
def handle_message(event: MessageEvent):
    user_id = get_user_id(event)
    user_text = (event.message.text or "").strip()

    # record user turn
    append_turn(user_id, "user", user_text if user_text else "(empty)")

    # generate with last 3 exchanges
    try:
        prompt = build_prompt(user_text, user_id)
        resp = model.generate_content(prompt)
        reply = (resp.text or "(No response)").strip()
        if len(reply) > 1000:
            reply = reply[:1000] + "…"
    except Exception as e:
        reply = f"Gemini error: {e}"

    # record assistant turn
    append_turn(user_id, "assistant", reply)

    # send reply
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                replyToken=event.reply_token,
                messages=[TextMessage(text=reply)]
            )
        )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT)
