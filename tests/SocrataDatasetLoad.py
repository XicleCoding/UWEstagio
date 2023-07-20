import pandas as pd
from sodapy import Socrata


LOAD_DATA = True


if LOAD_DATA :

    client = Socrata("data.sfgov.org", None)

    results = client.get("vw6y-z8j6", limit=100000)

    results_df = pd.DataFrame.from_records(results)


#print(results_df.head())
print(results_df[["service_name","service_subtype","service_details"]])

