import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from pydantic import BaseModel, Field, field_validator


class CustomerRecordSchema(BaseModel):
    """Pydantic schema enforcing data quality and type bounds for Telco subscriber records."""

    customerID: str
    Gender: str
    Tenure_Months: int = Field(..., alias="Tenure Months", ge=0, le=120)
    Monthly_Charges: float = Field(..., alias="Monthly Charges", ge=0.0, le=1000.0)
    Total_Charges: Optional[float] = Field(None, alias="Total Charges")
    Contract: str
    Churn_Label: Optional[int] = Field(None, alias="Churn Label")

    @field_validator("Gender")
    def validate_gender(cls, v: str) -> str:
        if v not in ["Male", "Female"]:
            raise ValueError(f"Invalid Gender value: {v}. Must be Male or Female.")
        return v

    @field_validator("Contract")
    def validate_contract(cls, v: str) -> str:
        valid_contracts = ["Month-to-month", "One year", "Two year"]
        if v not in valid_contracts:
            raise ValueError(f"Invalid Contract value: {v}. Must be one of {valid_contracts}.")
        return v


class DataValidator:
    """Validates Pandas DataFrames against Pydantic schema constraints."""

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def validate(self, df: pd.DataFrame) -> Tuple[bool, int, int]:
        if not self.enabled:
            return True, len(df), 0

        valid_count = 0
        invalid_count = 0
        errors = []

        records = df.to_dict(orient="records")
        for i, record in enumerate(records):
            try:
                # Replace string spaces with None for Total Charges
                if record.get("Total Charges") == " " or record.get("Total Charges") == "":
                    record["Total Charges"] = None
                elif record.get("Total Charges") is not None:
                    record["Total Charges"] = float(record["Total Charges"])
                
                CustomerRecordSchema.model_validate(record)
                valid_count += 1
            except Exception as e:
                invalid_count += 1
                errors.append(f"Row {i}: {e}")

        return valid_count > 0, valid_count, invalid_count
