from datetime import datetime, timedelta
import pandas as pd
import os
import logging
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.models import Variable
from time import sleep
from airflow.operators.email import EmailOperator

# Define the path to the local CSV file
LOCAL_CSV_FILE = '/home/growlt245/Downloads/Amended_Insurance_Product_purchase_dataset.csv'
BATCH_SIZE = 10000
S3_BUCKET_NAME = 'car-insurance-policy-bucket'
OUTPUT_FOLDER = 'car_insurance_data_raw'
BATCH_INTERVAL_SECONDS = 20

def upload_to_s3(batch_num, batch_data, s3_hook):
    try:
        # Get the timestamp for batch files
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")

        # Define the S3 object key with the folder path
        s3_key = f'{OUTPUT_FOLDER}/insurance_policy_data_batch_{batch_num}_{timestamp}.csv'
        
        # Save the batch data to a temporary CSV file
        temp_file = f'/tmp/car_policy_data_batch_{batch_num}_{timestamp}.csv'
        batch_data.to_csv(temp_file, index=False)
        
        # Upload the temporary CSV file to S3
        s3_hook.load_file(temp_file, key=s3_key, bucket_name=S3_BUCKET_NAME, replace=True)
        
        # Remove the temporary file
        os.remove(temp_file)
        logging.info(f'Batch {batch_num} uploaded successfully to {s3_key}')
    except Exception as e:
        logging.error(f'Error uploading batch {batch_num}: {e}')
        raise

def process_and_upload():
    try:
        # Print the current working directory
        print("Current working directory:", os.getcwd())
        
        # Read the entire CSV file
        df = pd.read_csv(LOCAL_CSV_FILE)
        
        # Retrieve the last processed batch number and last batch creation time from Airflow Variables
        last_batch_num = Variable.get("last_batch_num", default_var=0)
        last_batch_num = int(last_batch_num)
        
        last_batch_time = Variable.get("last_batch_time", default_var=str(datetime.min))
        last_batch_time = datetime.fromisoformat(last_batch_time)
        
        # Calculate the current batch number to be processed
        current_batch_num = last_batch_num + 1
        
        # Calculate the time elapsed since the last batch creation
        time_elapsed = datetime.now() - last_batch_time
        
        # If less than the specified interval has passed since the last batch, wait for the remaining time
        if time_elapsed < timedelta(seconds=BATCH_INTERVAL_SECONDS):
            sleep((timedelta(seconds=BATCH_INTERVAL_SECONDS) - time_elapsed).total_seconds())
        
        # Get unique customer IDs
        unique_customers = df['customer_ID'].unique()
        
        # Calculate the start and end indices for the current batch of unique customers
        start_idx = last_batch_num * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        
        # Get the batch of unique customer IDs
        batch_customer_ids = unique_customers[start_idx:end_idx]
        
        if len(batch_customer_ids) == 0:
            # Log that no new data was found
            logging.info("No new data found to process and upload.")
            return
        
        # Filter the dataframe for the current batch of unique customers
        batch_data = df[df['customer_ID'].isin(batch_customer_ids)]
        
        # Initialize the S3Hook
        s3_hook = S3Hook(aws_conn_id='local_to_s3')
        
        # Upload the current batch to S3
        upload_to_s3(current_batch_num, batch_data, s3_hook)
        
        # Update the last processed batch number and last batch creation time in Airflow Variables
        Variable.set("last_batch_num", current_batch_num)
        Variable.set("last_batch_time", datetime.now().isoformat())
    except Exception as e:
        logging.error(f'Error in process_and_upload: {e}')
        raise

# Define the default_args for the DAG
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2024, 7, 19),
    'email_on_failure': True,  # Set to False to prevent sending emails for every failure
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(seconds=10),
}

# Define the DAG
dag = DAG(
    'upload_a_batch_every_20_seconds',
    default_args=default_args,
    description='Extract customer data from a local CSV and upload to S3 in batches with a minimum 20 seconds interval between batches',
    schedule_interval=timedelta(seconds=20),  # Run the DAG every 20 seconds
    on_failure_callback=lambda context: notify_email(context),  # Set the failure callback here
)

# Define the EmailOperator task
notify_email_task = EmailOperator(
    task_id='send_email_on_failure',
    to=['zeel.gudhka@growexx.com'],
    subject='DAG Failed',
    html_content="""
    <h3>DAG Failure Alert</h3>
    <p>The DAG {{ dag.dag_id }} failed.</p>
    <p>Task: {{ task_instance.task_id }}</p>
    <p>Execution Date: {{ dag_run.execution_date }}</p>
    <p>Log URL: {{ task_instance.log_url }}</p>
    """,
    dag=dag,
)

# Define the task to process and upload data
upload_task = PythonOperator(
    task_id='process_and_upload_to_s3',
    python_callable=process_and_upload,
    dag=dag,
)

# Define task dependencies
upload_task
