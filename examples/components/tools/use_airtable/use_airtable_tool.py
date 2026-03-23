from dynamiq.connections import Airtable
from dynamiq.nodes.tools.airtable.bases import ListBases
from dynamiq.nodes.tools.airtable.records import ListRecords
from dynamiq.nodes.tools.airtable.views import ListViews

connection = Airtable(api_key="")


list_bases = ListBases(connection=connection)
list_records = ListRecords(connection=connection)
list_views = ListViews(connection=connection)

res = list_bases.run(input_data={})


print(res)

res_views = list_views.run(
    input_data={
        "base_id": "appfpQ3KHRlntsBol",
    }
)

print(res_views)


res_records = list_records.run(input_data={"base_id": "appgFz69gYPc8ewku", "table_id_or_name": "tblHeiBjXhsKW9Opj"})

print(res_records)
