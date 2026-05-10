from google.cloud import bigquery
import pandas as pd

class BigQueryHandler:
    def __init__(self, project_id):
        self.client = bigquery.Client(project=project_id)

    def get_training_data(self):
        # Using CTEs to aggregate user behavior
        query = """
        WITH user_sessions AS (
            SELECT 
                user_pseudo_id,
                COUNT(IF(event_name = 'session_start', 1, NULL)) as total_sessions,
                COUNT(IF(event_name = 'add_to_cart', 1, NULL)) as total_adds,
                MAX(event_timestamp) as last_active
            FROM `bigquery-public-data.ga4_obfuscated_sample_ecommerce.events_*`
            GROUP BY 1
        )
        SELECT * FROM user_sessions
        """
        # Using QueryJobConfig to manage resources
        job_config = bigquery.QueryJobConfig()
        query_job = self.client.query(query, job_config=job_config)
        return query_job.to_dataframe()
