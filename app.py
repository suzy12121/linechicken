# app.py
import os
from collections import deque
from typing import Dict, Deque, Tuple
from threading import Thread, Lock

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

# ---------- env & config ----------
load_dotenv()
CHANNEL_SECRET = os.getenv("LINE_CHANNEL_SECRET")
CHANNEL_ACCESS_TOKEN = os.getenv("LINE_CHANNEL_ACCESS_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
PORT = int(os.getenv("PORT", "5001"))
if not (CHANNEL_SECRET and CHANNEL_ACCESS_TOKEN and GOOGLE_API_KEY):
    raise RuntimeError("Missing one of LINE_CHANNEL_SECRET / LINE_CHANNEL_ACCESS_TOKEN / GOOGLE_API_KEY")

# LINE
handler = WebhookHandler(CHANNEL_SECRET)
configuration = Configuration(access_token=CHANNEL_ACCESS_TOKEN)

# Gemini
genai.configure(api_key=GOOGLE_API_KEY)
model = genai.GenerativeModel(
    model_name="gemini-1.5-flash",
    generation_config={"temperature": 0.6, "max_output_tokens": 256},
)

# ---------- short-term memory (last 3 exchanges) ----------
# history[user_id] = deque([("user","..."), ("assistant","..."), ...], maxlen=6)
history: Dict[str, Deque[Tuple[str, str]]] = {}
hist_lock = Lock()

def get_user_id(event: MessageEvent) -> str:
    src = getattr(event, "source", None)
    return getattr(src, "user_id", None) or getattr(src, "userId", None) or "anonymous"

def append_turn(user_id: str, role: str, text: str):
    with hist_lock:
        dq = history.get(user_id)
        if dq is None:
            dq = deque(maxlen=6)  # 3 exchanges (user+assistant)*3
            history[user_id] = dq
        dq.append((role, text))

def build_prompt(user_text: str, user_id: str) -> str:
    # gather history
    with hist_lock:
        past = list(history.get(user_id, deque()))
    lines = []
    for role, txt in past:
        speaker = "User" if role == "user" else "Assistant"
        lines.append(f"{speaker}: {txt}")
    past_text = "\n".join(lines) if lines else "（無對話記錄）"

    # system/style + knowledge, then recent convo
    prompt = f"""【Style Guide】 這是一個北一女中資訊研習社（北一資研）的吉祥物帳號，這隻吉祥物叫做「蘇格拉底雞」（英文名 Sugarlady）。至於為什麼叫蘇格拉底雞，仍然是個謎。
    請用友善的語氣回答問題，使用自然聊天的語言和格式，不要像「重點整理」。訊息不要太長，也不要使用粗體或斜體。你可以反問學妹問題，讓對話更自然。
    常用片語包含「很電」（＝厲害）、「破房」（＝完蛋、傷心）、「燒雞」（＝完蛋、沒救、失敗）。不要每個對話都推銷資研，也可以聊別的，
    但可以找有關聯的話題或是諧音梗連回資研。 
    【Knowledge Base】 北一女中資訊研習社（簡稱北一資研、北資）創社於民國 76 年 11 月 11 日，現任社師為楊喻文老師。
    北資有屬於社團獨立的 BBS ── 弗基斯特，大社課教授 C++，並鼓勵社員參與競賽，歷屆許多學姊表現十分優異。 
    中午和放學的額社秉持多元化傳統，致力提升校內資訊風氣，也舉辦活動聯絡感情。
    - 大社課：星期五的正式社課，由學術與社師教授不同主題（例如 C++）。 
    - 中午小社課：每天中午（12:10–12:40），由幹部教授課程，包括 C++、Python、AI、Unity、網頁、資安。 
    - 放學聯課：放學後由北資與建中電子計算機研習社（建電）幹部教授課程，包括 C++、Python、Unity、網頁、資安、機器學習。 
    友社：建電與北資自 1995 年合作，至今已傳承 30 屆，合稱建北電資。雙方關係密切，春夏秋冬的大活動和小社課都共同舉辦。 
    關於北一：北一女中有大熱食部（簡稱「大熱」）及小熱食部（簡稱「小熱」）。前者販售熟食、自助餐等，熱銷商品包含鍋燒意麵、草莓杯。
    後者販售點心、冰品及炸物為主，熱銷商品包含巧克力卡拉雞、薯不辣（薯條和甜不辣）等。北資的中午小社課使用電腦教室，不能吃午餐，但是會預留時間讓學妹下課後有空吃午餐。 
    這屆北資的幹部包含： 
    - 社長 Claire（克萊爾），人緣很好，是數資班大電神，負責資安的課程。 
    - 副社長兼公關 桃桃，很精緻可愛，喜歡烏薩奇，喜歡用豬鼻子的表情符號。據說臉很好捏。
    - 公關長 阿睿，很可愛又很帥，綽號貌似 array（陣列），很卷（認真唸書的意思）。 
    - 學術長 蘇西，負責北資的課程安排，是個興趣廣泛的生活白癡，也是這隻吉祥物的作者，負責 AI 和 Python 的課程。「蘇西」這個名字取自佩佩豬最好的朋友，小羊蘇西。 這支蘇格拉底雞是蘇西讓他可以講話的，但這隻蘇格拉底雞的角色是前幾屆學長屆創造並繪製出來的。
    - 學術 children（是 children 不是 child!），是社交達人，也是個大文豪，題目敘述和各種劇本經常都是她一手包辦，負責 Python 的課程。 
    - 學術 水餃，也是個數資大電神，看起來呆萌但其實超聰明，數學很好，喜歡抱著鯊鯊（建北電資的鯊魚娃娃），也是韓研（韓國文化研究社）的幹部，超電，負責網頁和 C++ 的課程。 
    - 學術兼文書 夜雨，超級 I 人（內向），感覺很傲嬌，負責遊戲製作軟體 Unity 的課程。 
    - 文書長 高高，溫柔姐姐，超級會畫畫。 也會跳舞！
    - 文書兼總務 小言，也是超級會畫畫，對陸劇集二次元日本動漫有研究。 
    - 雞蛋：雖然不是正式幹部，但是對北資貢獻很大！「雞蛋學姊是誰？」「雞蛋學姊是一顆雞蛋！」而且據說雞蛋學姊很容易暈船。她很電，喜歡看動漫，是個「超濃腐女」，喜歡看 BL。她喜歡聽音樂，包含日文、韓文、英文、中文歌， 然後會拉小提琴跟彈吉他，在建北電資聯合暑訓有彈吉他負責教小隊員唱歌喔！她也喜歡數學，自稱是個有氣質的小女孩。聽說蘇格拉底雞是她的姊姊。

    這屆建電的幹部包含：
    - 社長 國Llama 是個排球少年，數資班大電神。
    - 副社長 一拳 好像是個爛梗王，取名叫一拳據說是因為如果叫他一拳超人的話，他會給你一拳。是個盡責的人，扛。
    - 學術長 Eating（正在吃） 座位永遠很亂，是個情緒穩定的 chill guy。很常遲到，但其實超級電。不只資訊能力超電，他同時是建中班聯會的活動長，和航空社的幹部，十分不簡單。
    - 學術 Boron（硼砂）也是學術能力很電，喜歡玩音遊（音樂電動遊戲），常常玩到感覺鍵盤快被他打壞了。
    - 學術 Roy 競賽程式（競程）電神，做專案也很厲害！是個內向的人，但是跟熟的人會很吵
    - 學術兼網管 Benny班尼  保有學習的熱忱，是個外向的活潑男孩。喜歡講各種不可理喻的高深理論和幽默言詞。喜歡 Minecraft 模組開發、把 Rust 當腳本寫。
    - 學術兼網管 Windsor溫莎 號稱自己是「溫莎學姊」。現在負責重寫建北電資社網的工作，電。英文滿好的，講話微微抽象。代名詞請用「他」。
    - 衛生兼總務 Takora塔摳拉 喜歡吃奇怪的食物組合，例如水餃加番茄醬（建北電資很喜歡討論水餃到底可不可以加番茄醬的問題，水餃學姊表示不行）。在 instagram 上面超級活網，PO很多東西。同時是建中天文社的幹部，是個四幹人，電。
    - 總務 Brandon 雖然不是學術，但還是接了 Unity 的放學聯課，電。好像跟薑餅人有很大的關聯。
    - 文書兼公關 葉子 是建電裡面罕見的文組人，是李白狂粉，信手捻來就是文言文。會講相聲。會按讚每個人的每則限動。
    - 公關兼美宣 虛無裂德 大E人，很會社交。喜歡寶可夢，綽號是從寶可夢來的。會扮演死神。會寫 Scratch。

    9/12 的體驗社課是我們的學術小孩上的 Gemini API 入門唷，這週我們會更深入介紹 Gemini 的 API，並且帶大家進行 Python 的實作，地點在「至善 3 樓地理專科教室」。
    
    【Recent Conversation】(最多三輪) {past_text} 【User】 {user_text} 請在80字內回答問題。 
    
    
    """
    return prompt

# ---------- Flask app ----------
app = Flask(__name__)

@app.route("/", methods=["GET"])
def root_health():
    return "OK", 200

@app.route("/callback", methods=["POST", "GET"])
def callback():
    if request.method == "GET":
        return "OK", 200  # health check
    signature = request.headers.get("X-Line-Signature", "")
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400, "Invalid signature")
    # handler returns quickly because our message handler just spawns a thread
    return "OK", 200

def worker_reply(user_id: str, user_text: str, reply_token: str):
    # record user turn
    append_turn(user_id, "user", user_text or "(empty)")
    # call Gemini
    try:
        prompt = build_prompt(user_text, user_id)
        resp = model.generate_content(prompt)
        reply = (resp.text or "(No response)").strip()
        if len(reply) > 1000:
            reply = reply[:1000] + "…"
    except Exception as e:
        reply = f"Gemini error: {e}"
    # record bot turn
    append_turn(user_id, "assistant", reply)
    # reply to LINE
    with ApiClient(configuration) as api_client:
        MessagingApi(api_client).reply_message(
            ReplyMessageRequest(
                replyToken=reply_token,
                messages=[TextMessage(text=reply)]
            )
        )

@handler.add(MessageEvent, message=TextMessageContent)
def on_text_message(event: MessageEvent):
    user_id = get_user_id(event)
    user_text = (event.message.text or "").strip()
    reply_token = event.reply_token
    # spawn background worker so webhook returns immediately (avoid timeout)
    Thread(target=worker_reply, args=(user_id, user_text, reply_token), daemon=True).start()

if __name__ == "__main__":
    # Use a real WSGI server in production (e.g., gunicorn), but this is fine for local testing.
    app.run(host="0.0.0.0", port=PORT, threaded=True)
