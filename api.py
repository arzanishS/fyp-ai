import os

from flask import Flask
from flask_restful import reqparse
import predict as modelFile

app = Flask(__name__)


@app.route('/', methods=['GET'])
def hello_world():
    target = os.environ.get('TARGET', 'World')
    return "7"


@app.route('/price', methods=['POST'])
def loanPrediction():
    parser = reqparse.RequestParser()
    parser.add_argument('shipping')
    parser.add_argument('item_condition')
    parser.add_argument('brand_name')
    parser.add_argument('gen_cat')
    parser.add_argument('sub1_cat')
    parser.add_argument('sub2_cat')

    args = parser.parse_args()

    valueList = list(args.values())

    answer = modelFile.getPrediction(int(valueList[0]), int(valueList[1]), valueList[2], valueList[3], valueList[4],
                                     valueList[5])
    out = {'price': answer[0]}

    return out


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
