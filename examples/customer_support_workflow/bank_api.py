from flask import Flask, jsonify, request

app = Flask(__name__)

database = {
    "1234567890": {
        "pin_code": "912",
        "card_id": "1234567890",
        "account_id": "0011223344",
        "customer_name": "John Doe",
        "reason_for_blocking": "Suspicious activity detected",
        "blocked": True,
    },
    "1987654321": {
        "pin_code": "182",
        "account_id": "5566778899",
        "customer_name": "Jane Smith",
        "reason_for_blocking": "",
        "blocked": False,
    },
    "1122334455": {
        "pin_code": "756",
        "account_id": "7788990011",
        "customer_name": "Robert Johnson",
        "reason_for_blocking": "",
        "blocked": False,
    },
}


@app.route("/block_card", methods=["POST", "GET"])
def block_card():
    json_data = dict(request.form)
    card_number = str(json_data.get("card_number"))
    pin_code = str(json_data.get("pin_code"))
    if card_number not in database:
        return jsonify({"response": "Card number or pin code is invalid"})
    if database[card_number]["pin_code"] != pin_code:
        return jsonify({"response": "Card number or pin code is invalid, mismatch in pin codes"})
    if database[card_number]["blocked"]:
        return jsonify({"response": f"Card is already blocked. Reason {database[card_number]['reason_for_blocking']}"})
    database[card_number]["blocked"] = True
    return jsonify({"response": "Card was blocked successfully."})


# More endpoints

if __name__ == "__main__":
    app.run(port=5004)
