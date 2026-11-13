USE tempdb;
SET NOCOUNT ON;
CHECKPOINT;

SELECT
    name,
    CAST(size * 8.0 / 1024 AS decimal(12,2)) AS size_mb,
    physical_name
FROM sys.database_files
WHERE type_desc = 'LOG';

SELECT
    total_log_size_in_bytes / 1048576.0 AS total_log_mb,
    used_log_space_in_bytes / 1048576.0 AS used_log_mb,
    used_log_space_in_percent
FROM sys.dm_db_log_space_usage;

SELECT name, state_desc, recovery_model_desc, log_reuse_wait_desc
FROM sys.databases
WHERE database_id = DB_ID(N'tempdb');

SELECT COUNT_BIG(*) AS active_tempdb_user_transactions
FROM sys.dm_tran_database_transactions AS dt
JOIN sys.dm_tran_session_transactions AS st
    ON st.transaction_id = dt.transaction_id
WHERE dt.database_id = DB_ID(N'tempdb');

SELECT CASE WHEN OBJECT_ID(N'dbo.CA_TempdbLogGrowthLab', N'U') IS NULL
            THEN 'REMOVED - PASS'
            ELSE 'STILL EXISTS - CHECK'
       END AS lab_table_status;
