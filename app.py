import io
import json
import importlib
import pandas as pd
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import ValidationError

WRAPPER_MAPPER = {
    "jarir": "wrappers.jarir_wrapper",
    "cash": "wrappers.cash_wrapper",
    "purchase": "wrappers.purchase_wrapper",
}

app = FastAPI(
    title="Scoring API",
    description="Upload Excel, run model predictions, and download results",
)


@app.post("/predict")
async def predict(
    file: UploadFile,
    product: str = Form(default="jarir", description="Choose product: jarir | cash | purchase"),
    threshold: float = Form(default=0.3, description="Threshold between 0 and 1 (default: 0.3)"),
):
    # --- basic validation ---
    product = product.strip().lower()
    if product not in WRAPPER_MAPPER:
        raise HTTPException(status_code=400, detail=f"Unknown product '{product}'. Allowed: {list(WRAPPER_MAPPER.keys())}")

    if not (0 <= threshold <= 1):
        raise HTTPException(status_code=400, detail="Threshold must be between 0 and 1")

    if not file.filename.lower().endswith(".xlsx"):
        raise HTTPException(status_code=400, detail="Only .xlsx files are supported")

    # --- import wrapper ---
    try:
        wrapper_module = importlib.import_module(WRAPPER_MAPPER[product])
        ScoreModel = getattr(wrapper_module, "ScoreModel")
        InputData = getattr(wrapper_module, "InputData")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to import wrapper: {str(e)}")

    # --- read excel ---
    try:
        contents = await file.read()
        df = pd.read_excel(io.BytesIO(contents))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read Excel: {str(e)}")

    # --- get allowed fields from pydantic (v1/v2) ---
    if hasattr(InputData, "model_fields"):  # pydantic v2
        valid_fields = set(InputData.model_fields.keys())
    else:  # pydantic v1
        valid_fields = set(InputData.__fields__.keys())

    model = ScoreModel()
    proba_list, decision_list, shap_json_list = [], [], []

    for idx, row in df.iterrows():
        try:
            row_dict = row.to_dict()

            # drop any extra columns (including old proba/decision/shap_values)
            row_dict = {k: v for k, v in row_dict.items() if k in valid_fields}

            # special case for cash
            if "occupationcode" in row_dict:
                row_dict["occupationcode"] = str(row_dict.get("occupationcode", ""))

            input_data = InputData(**row_dict)
            _, result = model.predict(input_data, threshold=threshold)

            proba_list.append(result.get("probability_of_default"))
            decision_list.append(result.get("decision"))
            shap_json_list.append(json.dumps(result.get("shap_values"), ensure_ascii=False))
        except ValidationError as e:
            # показываем, на какой строке упали
            raise HTTPException(status_code=422, detail=f"Validation error in row {idx}: {e}")
        except Exception as e:
            # если внезапно что-то ещё сломалось — не роняем весь файл
            proba_list.append(None)
            decision_list.append(None)
            shap_json_list.append(None)

    df["proba"] = proba_list
    df["decision"] = decision_list
    df["shap_values"] = shap_json_list

    # --- return excel ---
    output = io.BytesIO()
    df.to_excel(output, index=False)
    output.seek(0)
    filename = file.filename.replace(".xlsx", "_predictions.xlsx")

    return StreamingResponse(
        output,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": f"attachment; filename={filename}"}
    )
