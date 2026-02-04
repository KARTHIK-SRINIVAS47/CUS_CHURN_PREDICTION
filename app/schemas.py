
from pydantic import BaseModel


class CustomerData(BaseModel):
    # Numeric features
    Latitude: float
    Longitude: float
    Tenure_Months: int
    Monthly_Charges: float
    Total_Charges: float

    # Categorical features
    State: str
    Gender: str
    Senior_Citizen: str
    Partner: str
    Dependents: str
    Phone_Service: str
    Multiple_Lines: str
    Internet_Service: str
    Online_Security: str
    Online_Backup: str
    Device_Protection: str
    Tech_Support: str
    Streaming_TV: str
    Streaming_Movies: str
    Contract: str
    Paperless_Billing: str
    Payment_Method: str
