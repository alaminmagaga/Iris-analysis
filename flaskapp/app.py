from flask import Flask, render_template, request, redirect, url_for

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/result', methods=['POST'])
def result():
    num1 = float(request.form['num1'])
    num2 = float(request.form['num2'])
    result = num1 + num2
    operation = request.form['operation']
    if operation == 'addition':
        result = num1 + num2
    elif operation == 'division':
        if num2 != 0:
            result = num1 / num2
        else:
            result = "Error: Division by zero"

    return render_template('result.html', result=result)






if __name__ == '__main__':
    app.run(debug=True)
