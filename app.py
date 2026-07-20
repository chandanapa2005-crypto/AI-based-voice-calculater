import re
import math
import webbrowser
from threading import Timer
from collections import Counter, defaultdict
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

STATES = (
    "ADD", "SUBTRACT", "MULTIPLY", "DIVIDE", "FLOOR_DIVIDE",
    "MODULUS", "POWER", "SQRT", "FACTORIAL", "PERCENTAGE",
    "ABSOLUTE", "SIN", "COS", "TAN", "COT", "SEC", "COSEC"
)

TRAINING_DATA = {
    "ADD":"add plus sum addition",
    "SUBTRACT":"subtract minus difference",
    "MULTIPLY":"multiply multiplied times product",
    "DIVIDE":"divide divided division quotient",
    "FLOOR_DIVIDE":"floor divide integer division",
    "MODULUS":"mod modulo modulus remainder",
    "POWER":"power raised exponent",
    "SQRT":"square root sqrt",
    "FACTORIAL":"factorial",
    "PERCENTAGE":"percent percentage",
    "ABSOLUTE":"absolute value abs",
    "SIN":"sin sine", "COS":"cos cosine",
    "TAN":"tan tangent", "COT":"cot cotangent",
    "SEC":"sec secant", "COSEC":"cosec cosecant"
}

OPERATION_WORDS = set("""
add plus sum addition subtract minus difference
multiply multiplied times product divide divided division quotient
floor integer mod modulo modulus remainder power raised exponent
square root sqrt factorial percent percentage absolute value abs
sin sine cos cosine tan tangent cot cotangent sec secant cosec cosecant
""".split())

NUMBER_WORDS = {
    "zero":0, "one":1, "two":2, "three":3, "four":4, "five":5,
    "six":6, "seven":7, "eight":8, "nine":9, "ten":10,
    "eleven":11, "twelve":12, "thirteen":13, "fourteen":14,
    "fifteen":15, "sixteen":16, "seventeen":17, "eighteen":18,
    "nineteen":19, "twenty":20, "thirty":30, "forty":40,
    "fifty":50, "sixty":60, "seventy":70, "eighty":80, "ninety":90
}

def normalise(text):
    text = str(text).lower().strip()
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%\s*of\b", r"\1 percent of", text)
    text = re.sub(r"(\d+(?:\.\d+)?)\s*%(?!\s*\d)", r"\1 percent", text)
    text = re.sub(r"(?<=\d)\s*-\s*(?=\d)", " minus ", text)

    text = (
        text.replace("//", " floor divide ")
        .replace("÷", " divided by ")
        .replace("/", " divided by ")
        .replace("×", " times ")
        .replace("*", " times ")
        .replace("+", " plus ")
        .replace("^", " power ")
        .replace("%", " modulo ")
        .replace("√", " square root ")
        .replace("!", " factorial ")
    )

    for wrong, correct in {
        "tree":"three", "free":"three",
        "madulo":"modulo", "module":"modulo"
    }.items():
        text = re.sub(rf"\b{wrong}\b", correct, text)

    return text


class HMMCalculator:
    def __init__(self):
        self.emissions = defaultdict(Counter)
        self.start_counts = Counter()
        self.transitions = defaultdict(Counter)
        self.vocabulary = set()
        self.train()

    def operation_words(self, text):
        return [
            word for word in re.findall(r"[a-z]+", normalise(text))
            if word in OPERATION_WORDS
        ]

    def train(self):
        for state, sentence in TRAINING_DATA.items():
            words = self.operation_words(sentence)
            self.start_counts[state] += 1

            for word in words:
                self.emissions[state][word] += 1
                self.vocabulary.add(word)

            for _ in range(len(words) - 1):
                self.transitions[state][state] += 1

    def start_probability(self, state):
        return (self.start_counts[state] + 1) / (
            sum(self.start_counts.values()) + len(STATES)
        )

    def emission_probability(self, word, state):
        alpha = 0.1
        return (self.emissions[state][word] + alpha) / (
            sum(self.emissions[state].values())
            + alpha * len(self.vocabulary)
        )

    def transition_probability(self, old, new):
        alpha = 0.1
        return (self.transitions[old][new] + alpha) / (
            sum(self.transitions[old].values())
            + alpha * len(STATES)
        )

    def viterbi_predict(self, text):
        words = self.operation_words(text)

        if not words:
            return None

        scores = {
            state: math.log(self.start_probability(state))
            + math.log(self.emission_probability(words[0], state))
            for state in STATES
        }

        for word in words[1:]:
            scores = {
                new: max(
                    scores[old]
                    + math.log(self.transition_probability(old, new))
                    + math.log(self.emission_probability(word, new))
                    for old in STATES
                )
                for new in STATES
            }

        return max(scores, key=scores.get)

    def get_numbers(self, text):
        values = []
        current = None
        negative = False

        for token in re.findall(
            r"-?\d+(?:\.\d+)?|[a-z]+",
            normalise(text)
        ):
            if re.fullmatch(r"-?\d+(?:\.\d+)?", token):
                if current is not None:
                    values.append(float(-current if negative else current))
                    current, negative = None, False

                values.append(float(token))

            elif token == "negative":
                negative = True

            elif token in NUMBER_WORDS:
                current = (current or 0) + NUMBER_WORDS[token]

            elif token == "hundred" and current is not None:
                current *= 100

            elif current is not None:
                values.append(float(-current if negative else current))
                current, negative = None, False

        if current is not None:
            values.append(float(-current if negative else current))

        return values

    def calculate(self, text):
        operation = self.viterbi_predict(text)
        numbers = self.get_numbers(text)
        clean = normalise(text)

        if operation is None:
            return "I could not detect an operation."

        if operation == "SQRT":
            if not numbers or numbers[0] < 0:
                return "Square root needs a non-negative number."

            answer = math.sqrt(numbers[0])
            expression = f"Square root of {numbers[0]:g}"

        elif operation == "FACTORIAL":
            if not numbers or numbers[0] < 0 or not numbers[0].is_integer():
                return "Factorial needs a non-negative whole number."

            answer = math.factorial(int(numbers[0]))
            expression = f"{numbers[0]:g}!"

        elif operation == "ABSOLUTE":
            if not numbers:
                return "Please provide a number."

            answer = abs(numbers[0])
            expression = f"Absolute value of {numbers[0]:g}"

        elif operation == "PERCENTAGE":
            if not numbers:
                return "Please provide a percentage."

            answer = numbers[0] / 100 if len(numbers) == 1 else numbers[0] / 100 * numbers[1]
            expression = (
                f"{numbers[0]:g}%"
                if len(numbers) == 1
                else f"{numbers[0]:g}% of {numbers[1]:g}"
            )

        elif operation in ("SIN", "COS", "TAN", "COT", "SEC", "COSEC"):
            if not numbers:
                return "Please provide an angle in degrees."

            angle = numbers[0]
            sine = math.sin(math.radians(angle))
            cosine = math.cos(math.radians(angle))

            if operation == "SIN":
                answer, name = sine, "sin"
            elif operation == "COS":
                answer, name = cosine, "cos"
            elif operation == "TAN":
                if abs(cosine) < 0.0000001:
                    return "Tangent is undefined for this angle."
                answer, name = math.tan(math.radians(angle)), "tan"
            elif operation == "COT":
                if abs(sine) < 0.0000001:
                    return "Cotangent is undefined for this angle."
                answer, name = cosine / sine, "cot"
            elif operation == "SEC":
                if abs(cosine) < 0.0000001:
                    return "Secant is undefined for this angle."
                answer, name = 1 / cosine, "sec"
            else:
                if abs(sine) < 0.0000001:
                    return "Cosecant is undefined for this angle."
                answer, name = 1 / sine, "cosec"

            expression = f"{name}({angle:g} degrees)"

        else:
            if len(numbers) < 2:
                return "Please provide two numbers."

            first, second = numbers[0], numbers[1]

            if operation == "SUBTRACT" and "from" in clean:
                first, second = second, first

            if operation == "ADD":
                answer, symbol = first + second, "+"
            elif operation == "SUBTRACT":
                answer, symbol = first - second, "-"
            elif operation == "MULTIPLY":
                answer, symbol = first * second, "×"
            elif operation == "DIVIDE":
                if second == 0:
                    return "Division by zero is not allowed."
                answer, symbol = first / second, "÷"
            elif operation == "FLOOR_DIVIDE":
                if second == 0:
                    return "Division by zero is not allowed."
                answer, symbol = first // second, "//"
            elif operation == "MODULUS":
                if second == 0:
                    return "Modulus by zero is not allowed."
                answer, symbol = first % second, "%"
            else:
                answer, symbol = first ** second, "^"

            expression = f"{first:g} {symbol} {second:g}"

        return f"HMM predicted: {operation}. {expression} = {answer:g}. The answer is {answer:g}."


calculator = HMMCalculator()
history = []

HTML = """
<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>AI Voice Calculator</title>
<style>
body{margin:0;background:#10182b;color:#eef2ff;font-family:Arial;padding:20px}
.box{width:min(100%,540px);margin:auto;background:#111827;padding:25px;border-radius:25px}
header{display:flex;justify-content:space-between;margin-bottom:20px}
h1{margin:0;font-size:23px}.tag{color:#c9c1ff}
.screen{height:145px;background:#080e1c;border-radius:20px;padding:22px;display:flex;flex-direction:column;justify-content:end;align-items:end}
input{width:100%;background:transparent;border:0;outline:0;color:#c3cdea;text-align:right;font:18px monospace}
#answer{font-size:36px;font-weight:bold;margin-top:10px}
#message{color:#bac4df;min-height:24px}
.grid{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin-bottom:8px}
.keys{grid-template-columns:repeat(4,1fr)}
button{min-height:58px;border:0;border-radius:15px;background:#293653;color:white;font-weight:bold;cursor:pointer}
.grid button{min-height:45px}.op{background:#393663}.equal,.history-show{background:#7864ff}
.clear,.history-clear{background:#613244;color:#ffc9cf}.wide{grid-column:span 2}
#history{margin-top:14px;padding:12px;max-height:180px;overflow:auto;background:#080e1c;border-radius:12px;font-size:13px}
.item{padding:8px 0;border-bottom:1px solid #ffffff18}
</style>
</head>
<body>
<div class="box">
<header><h1>AI Voice Calculator</h1><span class="tag">HMM-style · Scientific</span></header>

<div class="screen">
<input id="input" placeholder="Example: add 10 and 20">
<div id="answer">0</div>
</div>

<p id="message">Type or use calculator buttons.</p>

<div class="grid">
<button onclick="add('sin ')">sin</button><button onclick="add('cos ')">cos</button><button onclick="add('tan ')">tan</button><button onclick="add('cot ')">cot</button><button onclick="add('sec ')">sec</button>
<button onclick="add('cosec ')">cosec</button><button onclick="add('square root ')">√</button><button onclick="add('factorial ')">x!</button><button onclick="add('absolute value ')">abs</button><button onclick="voice()">🎙</button>
<button onclick="add(' power ')">xʸ</button><button onclick="add(' modulo ')">mod</button><button onclick="add(' floor divide ')">⌊/⌋</button><button onclick="add(' percent ')">%</button><button onclick="add(' ')">space</button>
</div>

<div class="grid keys">
<button class="clear" onclick="clearInput()">AC</button><button onclick="back()">⌫</button><button onclick="add('(')">(</button><button onclick="add(')')">)</button>
<button onclick="add('7')">7</button><button onclick="add('8')">8</button><button onclick="add('9')">9</button><button class="op" onclick="add(' divided by ')">÷</button>
<button onclick="add('4')">4</button><button onclick="add('5')">5</button><button onclick="add('6')">6</button><button class="op" onclick="add(' times ')">×</button>
<button onclick="add('1')">1</button><button onclick="add('2')">2</button><button onclick="add('3')">3</button><button class="op" onclick="add(' minus ')">−</button>
<button class="wide" onclick="add('0')">0</button><button onclick="add('.')">.</button><button class="op" onclick="add(' plus ')">+</button>

<button class="equal wide" onclick="calculate()">=</button>
<button class="history-show" onclick="showHistory()">History</button>
<button class="history-clear" onclick="clearHistory()">Clear</button>
</div>

<div id="history">No calculations yet.</div>
</div>

<script>
const input=document.getElementById("input");
const answer=document.getElementById("answer");
const message=document.getElementById("message");

function add(x){input.value+=x;input.focus()}
function back(){input.value=input.value.slice(0,-1);input.focus()}
function clearInput(){input.value="";answer.textContent="0";message.textContent="Cleared."}
function speak(x){speechSynthesis.cancel();speechSynthesis.speak(new SpeechSynthesisUtterance(x))}

async function calculate(){
  if(!input.value.trim())return;

  const response=await fetch("/calculate",{
    method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({text:input.value})
  });

  const data=await response.json();
  message.textContent=data.result;
  answer.textContent=data.result.match(/The answer is ([^.]+)\./)?.[1]||"Done";
  speak(data.result);
  loadHistory();
}

async function loadHistory(){
  const response=await fetch("/history");
  const items=await response.json();
  const box=document.getElementById("history");

  box.innerHTML=items.length
    ? items.map(x=>`<div class="item"><b>${x.input}</b><br>${x.result}</div>`).join("")
    : "No calculations yet.";
}

async function showHistory(){
  await loadHistory();
  document.getElementById("history").scrollIntoView({
    behavior:"smooth",
    block:"center"
  });
}

async function clearHistory(){
  await fetch("/clear-history",{method:"POST"});
  await loadHistory();
  message.textContent="Calculation history cleared.";
}

function voice(){
  const R=window.SpeechRecognition||window.webkitSpeechRecognition;

  if(!R){
    message.textContent="Open in Chrome or Edge for voice input.";
    return;
  }

  const r=new R();
  r.lang="en-IN";

  r.onstart=()=>message.textContent="Listening...";

  r.onresult=e=>{
    const text=e.results[0][0].transcript.toLowerCase().trim();

    if(text.includes("show history")){
      showHistory();
      return;
    }

    if(text.includes("clear history")){
      clearHistory();
      return;
    }

    input.value=text;
    calculate();
  };

  r.onerror=()=>message.textContent="Voice not understood.";
  r.start();
}

input.onkeydown=e=>{if(e.key==="Enter")calculate()}
loadHistory();
</script>
</body>
</html>
"""

@app.get("/")
def home():
    return render_template_string(HTML)

@app.post("/calculate")
def calculate_web():
    text = request.get_json().get("text", "")
    result = calculator.calculate(text)
    history.insert(0, {"input": text, "result": result})
    return jsonify({"result": result})

@app.get("/history")
def get_history():
    return jsonify(history[:10])

@app.post("/clear-history")
def clear_history():
    history.clear()
    return jsonify({"message": "History cleared"})

if __name__ == "__main__":
    Timer(1, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)