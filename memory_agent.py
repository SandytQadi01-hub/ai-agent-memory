# -*- coding: utf-8 -*-
# Agent كامل: ذاكرة (Class) + تفكير متعدد الخطوات (ReAct) + معالجة أخطاء.
import os
import difflib
import requests
import pandas as pd

MODEL = "llama3.2:3b"
GEN_URL = "http://localhost:11434/api/generate"
BASE = os.path.dirname(os.path.abspath(__file__))


# ============================================================
# الجزء 1 — نظام الذاكرة كـ Class
# ============================================================

class ConversationMemory:
    """ذاكرة منظّمة تخزّن رسائل المستخدم وردود النموذج ونتائج الأدوات."""

    def __init__(self):
        self.history = []

    # تسجّل رسالة من المستخدم.
    def add_user_message(self, text):
        self.history.append(("user", text))

    # تسجّل ردًّا من النموذج (تفكير/فعل/جواب).
    def add_assistant_message(self, text):
        self.history.append(("assistant", text))

    # تسجّل نتيجة تنفيذ أداة.
    def add_tool_result(self, tool_name, result):
        self.history.append(("tool", f"[{tool_name}] {result}"))

    # ترجع كل الذاكرة كنص واحد منسّق جاهز للـ Prompt.
    def get_context_string(self):
        lines = []
        for role, content in self.history:
            if role == "user":
                lines.append(f"User: {content}")
            elif role == "assistant":
                lines.append(f"Assistant: {content}")
            elif role == "tool":
                lines.append(f"Observation: {content}")
        return "\n".join(lines)


# ============================================================
# الأدوات الثلاث
# ============================================================

# تنفّذ عملية حسابية نصية وتعيد الناتج أو رسالة خطأ واضحة.
def calculator_tool(expression):
    allowed = "0123456789+-*/(). %"
    expr = "".join(ch for ch in expression if ch in allowed).strip()
    try:
        return str(eval(expr, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"خطأ بالعملية الحسابية: {e}"


# تبحث عن كلمة مفتاحية في company_info.txt وتعيد الأسطر المطابقة.
def search_file_tool(keyword):
    try:
        with open(os.path.join(BASE, "company_info.txt"), encoding="utf-8") as f:
            lines = f.readlines()
    except FileNotFoundError:
        return "الملف غير موجود."
    key = keyword.strip().lower()
    matches = [ln.strip() for ln in lines if key in ln.lower()]
    return "\n".join(matches) if matches else "ما لقيت معلومة."


# تبحث عن منتج (بحث تقريبي) في products.csv وتعيد سعره وكميته أو رسالة عدم وجود.
def read_data_tool(product_name):
    try:
        df = pd.read_csv(os.path.join(BASE, "products.csv"))
    except FileNotFoundError:
        return "ملف المنتجات غير موجود."
    query = product_name.strip().lower()
    names = df["product_name"].astype(str).tolist()
    idx = None
    for i, name in enumerate(names):
        if query in name.lower() or name.lower() in query:
            idx = i
            break
    if idx is None:
        close = difflib.get_close_matches(query, [n.lower() for n in names], n=1, cutoff=0.75)
        if close:
            idx = [n.lower() for n in names].index(close[0])
    if idx is None:
        return f"المنتج '{product_name}' غير موجود."
    row = df.iloc[idx]
    return f"سعر {row['product_name']} = {row['price']}، الكمية المتوفرة = {row['quantity_in_stock']}"


TOOLS = {"CALCULATOR": calculator_tool, "SEARCH_FILE": search_file_tool, "READ_DATA": read_data_tool}


# ============================================================
# الجزء 2 — التفكير متعدد الخطوات (ReAct)
# ============================================================

SYSTEM_PROMPT = (
    "أنت مساعد لديك ثلاث أدوات:\n"
    "- CALCULATOR: لأي عملية حسابية. مثال: CALCULATOR: 12*3\n"
    "- SEARCH_FILE: للبحث عن معلومة عن الشركة. مثال: SEARCH_FILE: المقر\n"
    "- READ_DATA: لمعرفة سعر/كمية منتج. مثال: READ_DATA: لابتوب\n\n"
    "حل السؤال خطوة بخطوة. في كل ردّ اكتب إمّا خطوة أداة واحدة بهذا الشكل:\n"
    "Thought: <تفكيرك المختصر>\n"
    "Action: TOOL_NAME: <المدخل>\n"
    "ثم توقّف وانتظر سطر Observation (لا تكتب Observation بنفسك).\n"
    "أو عندما تعرف الجواب اكتب سطرًا واحدًا:\n"
    "Final Answer: <الجواب بالعربية>\n"
    "قواعد مهمة: استخدم أداة واحدة فقط في كل خطوة، ولا تخترع أرقامًا — إن احتجت سعر "
    "منتج فاستخدم READ_DATA أولًا ثم CALCULATOR. لا تكرّر خطوة نفّذتها، بل استعمل "
    "نتيجة الـ Observation.\n\n"
    "مثال لسؤال من خطوتين: «كم سعر الشاشة مضروب بـ2؟»\n"
    "Thought: أحتاج أولًا سعر الشاشة.\n"
    "Action: READ_DATA: شاشة\n"
    "Observation: سعر شاشة = 180، الكمية المتوفرة = 30\n"
    "Thought: الآن أضرب 180 في 2.\n"
    "Action: CALCULATOR: 180*2\n"
    "Observation: 360\n"
    "Final Answer: سعر الشاشة مضروب بـ2 هو 360."
)


# يرسل prompt للنموذج ويعيد النص (يتوقف افتراضيًا قبل أن يخترع Observation).
def _generate(prompt, stop=("Observation:",)):
    r = requests.post(GEN_URL, json={"model": MODEL, "prompt": prompt, "stream": False,
                                     "options": {"temperature": 0, "stop": list(stop)}}, timeout=180)
    return r.json()["response"]


# يجبر النموذج على كتابة الجواب النهائي مستخدمًا نتائج الـ Observations المتوفرة.
def _force_final(memory):
    prompt = (SYSTEM_PROMPT + "\n\n" + memory.get_context_string() +
              "\nلديك كل المعلومات من الـ Observations أعلاه. اكتب الآن سطرًا واحدًا فقط "
              "يبدأ بـ 'Final Answer:' ويجيب سؤال المستخدم بالعربية.")
    text = _generate(prompt, stop=("\n",)).strip()
    return _extract_final(text) or text


# يستخرج (اسم الأداة، المدخل) من سطر "Action:" فقط (وليس من سطر Thought).
def _parse_action(text):
    for line in text.splitlines():
        if "Action:" in line:
            after = line.split("Action:", 1)[1]
            for name in TOOLS:
                if name in after:
                    tool_input = after.split(name, 1)[1].lstrip(": ").strip()
                    return name, tool_input
    return None, None


# يستخرج نص ما بعد "Final Answer:".
def _extract_final(text):
    marker = "Final Answer:"
    idx = text.find(marker)
    if idx == -1:
        return None
    return text[idx + len(marker):].strip().splitlines()[0].strip()


# الحلقة متعددة الخطوات: Thought -> Action -> Observation -> ... -> Final Answer.
def run_agent(user_question, memory, max_steps=5, verbose=True):
    memory.add_user_message(user_question)
    reformat_used = False
    done_actions = {}
    for step in range(max_steps):
        prompt = SYSTEM_PROMPT + "\n\n" + memory.get_context_string() + "\nاكتب الخطوة التالية:\n"
        text = _generate(prompt).strip()
        if verbose:
            print(f"  [خطوة {step + 1}] {text}")

        final = _extract_final(text)
        tool, tool_input = _parse_action(text)

        # الحالة 3 من معالجة الأخطاء: صيغة غير معروفة -> اطلب إعادة الصياغة مرة
        if final is None and tool is None:
            if not reformat_used:
                reformat_used = True
                memory.add_assistant_message(text)
                memory.add_tool_result("SYSTEM",
                                       "الرد غير مفهوم. أعد الكتابة بالصيغة: Action: TOOL: input أو Final Answer: ...")
                if verbose:
                    print("  [معالجة خطأ] صيغة غير معروفة -> طلب إعادة الصياغة")
                continue
            answer = "لم أتمكن من فهم الطلب بصيغة صحيحة."
            memory.add_assistant_message(answer)
            return answer

        # الأولوية للفعل (استخدام أداة) قبل الجواب النهائي
        if tool is not None:
            key = (tool, tool_input)
            if done_actions.get(key, 0) >= 1:
                # النموذج يكرر خطوة نُفّذت -> أجبره على كتابة الجواب النهائي من النتائج المتوفرة
                if verbose:
                    print("  [معالجة خطأ] تكرار خطوة -> إجبار على الجواب النهائي")
                answer = _force_final(memory)
                memory.add_assistant_message(f"Final Answer: {answer}")
                return answer
            memory.add_assistant_message(text)
            observation = TOOLS[tool](tool_input)
            memory.add_tool_result(tool, observation)
            done_actions[key] = done_actions.get(key, 0) + 1
            if verbose:
                print(f"  [الأداة] {tool}({tool_input}) = {observation}")
            continue

        if final is not None:
            memory.add_assistant_message(f"Final Answer: {final}")
            return final

    # الحالة 3: حماية من التكرار اللانهائي
    msg = "لم أصل إلى جواب مؤكد ضمن عدد الخطوات المسموح."
    memory.add_assistant_message(msg)
    return msg


# ============================================================
# الاختبارات
# ============================================================

def demo_memory_class():
    print("########## الجزء 1: اختبار ConversationMemory ##########")
    m = ConversationMemory()
    m.add_user_message("ما هي عاصمة الأردن؟")
    m.add_assistant_message("عمّان.")
    m.add_user_message("وكم عدد سكانها؟")
    m.add_assistant_message("حوالي 4 ملايين.")
    print(m.get_context_string())
    print("عدد العناصر في الذاكرة:", len(m.history))


def demo_multi_step():
    print("\n########## الجزء 2: تفكير متعدد الخطوات ##########")
    tests = [
        "كم سعر اللابتوب مضروب بـ3؟",
        "كم مجموع سعر الماوس وسعر الكيبورد؟",
        "كم تصير قيمة كل قطع اللابتوب المتوفرة (السعر × الكمية)؟",
    ]
    for i, q in enumerate(tests, 1):
        print(f"\n--- سؤال {i}: {q} ---")
        answer = run_agent(q, ConversationMemory())
        print(f"  >> Final Answer: {answer}")


def demo_error_handling():
    print("\n########## الجزء 3: معالجة الأخطاء ##########")
    print("\n[حالة 2] أداة ترجع خطأ (منتج غير موجود) ثم حساب:")
    a2 = run_agent("كم سعر الطائرة مضروب بـ2؟", ConversationMemory())
    print(f"  >> {a2}")

    print("\n[حالة 3] سؤال مربك لا علاقة له بأي أداة (حماية من التكرار):")
    a3 = run_agent("احكيلي قصة عن التنين الأزرق الطائر؟", ConversationMemory(), max_steps=3)
    print(f"  >> {a3}")


# الجزء 4 — محادثة متكاملة واحدة بذاكرة مشتركة (اختبار السيناريو الكامل).
def demo_full_conversation():
    print("\n########## الجزء 4: محادثة متكاملة (ذاكرة مشتركة) ##########")
    memory = ConversationMemory()
    conversation = [
        "كم سعر اللابتوب؟",                              # 1) أداة واحدة
        "واضربه بـ2 كم يصير؟",                           # 2) يعتمد على السياق (الذاكرة)
        "كم مجموع سعر الماوس والكيبورد؟",                # 3) أكثر من أداة بالتسلسل
        "كم سعر الطائرة؟",                               # 4) خطأ متعمّد (غير موجود)
        "وين مقر الشركة؟",                               # 5) أداة البحث
        "ذكّرني: شو كان سعر اللابتوب اللي سألت عنه بالبداية؟",  # 6) ذاكرة عميقة
    ]
    for i, q in enumerate(conversation, 1):
        print(f"\n[رسالة {i}] المستخدم: {q}")
        answer = run_agent(q, memory, verbose=False)
        print(f"  الـ Agent: {answer}")


if __name__ == "__main__":
    demo_memory_class()
    demo_multi_step()
    demo_error_handling()
    demo_full_conversation()
