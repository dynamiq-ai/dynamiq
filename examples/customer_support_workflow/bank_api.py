import uvicorn
from fastapi import FastAPI, Form

app = FastAPI()


database = {
    "1234567890": {
        "pin_code": "912",
        "card_id": "1234567890",
        "customer_name": "John Doe",
        "blocked": True,
    },
    "1987654321": {
        "pin_code": "182",
        "customer_name": "Jane Smith",
        "blocked": False,
    },
    "1122334455": {
        "pin_code": "756",
        "customer_name": "Robert Johnson",
        "blocked": False,
    },
}


@app.post("/block_card")
def block_card(card_number: str = Form(...), pin_code: str = Form(...)):
    if card_number not in database:
        return {"response": "Card number or pin code is invalid."}
    if database[card_number]["pin_code"] != pin_code:
        return {"response": "Card number or pin code is invalid, mismatch in pin codes."}
    if database[card_number]["blocked"]:
        return {"response": "Card is already blocked."}
    database[card_number]["blocked"] = True
    return {"response": "Card was blocked successfully."}


@app.post("/make_transaction")
def block_card(card_number_sender: str = Form(...), card_number_reciever: str = Form(...), amount: int = Form(...)):
    if card_number_reciever in database and card_number_sender in database:
        return {"response": f"Successfully transferred {amount} dollars to the receiver."}
    return {"response": "Invalid credentials."}


@app.post("/request_report")
def block_card(card_number: str = Form(...), pin_code: str = Form(...)):
    account_details = database.get(card_number)
    if account_details and account_details.get("pin_code") == pin_code:
        return {
            "response": (f"Username: {account_details['customer_name']}."
                          "Account status: {'blocked' if account_details['blocked'] else 'active'}.")}
    return {"response": "Invalid credentials."}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
