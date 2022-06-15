from azureml.core import Workspace
from azureml.core import Datastore
import json
import argparse


def prepare(workspace_name, resource_group, subscription_id):
    print(workspace_name)
    ws = Workspace(subscription_id=subscription_id, resource_group=resource_group, workspace_name=workspace_name)
    ws.write_config(file_name=f"{workspace_name}.json")

    if datastore_name not in ws.datastores:
        Datastore.register_azure_blob_container(
            workspace=ws,
            datastore_name=datastore_name,
            container_name=blob_container_name,
            account_name=blob_account_name,
            account_key=blob_account_key
        )
        print("Datastore '%s' registered." % datastore_name)
    else:
        print("Datastore '%s' has already been regsitered." % datastore_name)

    print("Success!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AzureML Prepare')
    parser.add_argument(
        '--azureml_config_file', 
        required=False, 
        default='run_on_aml/v-kahu1.json',
    )
    args = parser.parse_args()
    with open(args.azureml_config_file, "r") as fp:
        data = json.load(fp)
    workspaces = data['workspaces']
    datastore_name = data['datastore_name']
    blob_container_name = data['blob_container_name']
    blob_account_name = data['blob_account_name']
    blob_account_key = data['blob_account_key']
    for workspace_name, workspace_data in workspaces.items():
        resource_group = workspace_data['resource_group']
        subscription_id = workspace_data['subscription_id']
        prepare(workspace_name, resource_group, subscription_id)