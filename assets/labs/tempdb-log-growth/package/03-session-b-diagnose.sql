USE tempdb;
SET NOCOUNT ON;

SELECT
    name,
    CAST(size * 8.0 / 1024 AS decimal(12,2)) AS size_mb,
    CASE WHEN is_percent_growth = 1 THEN CONCAT(growth, '%')
         ELSE CONCAT(CAST(growth * 8.0 / 1024 AS decimal(12,2)), ' MB')
    END AS growth_setting,
    physical_name
FROM sys.database_files
WHERE type_desc = 'LOG';

SELECT
    total_log_size_in_bytes / 1048576.0 AS total_log_mb,
    used_log_space_in_bytes / 1048576.0 AS used_log_mb,
    used_log_space_in_percent
FROM sys.dm_db_log_space_usage;

SELECT name, state_desc, log_reuse_wait_desc
FROM sys.databases
WHERE database_id = DB_ID(N'tempdb');

SELECT
    at.transaction_id,
    at.name AS transaction_name,
    at.transaction_begin_time,
    dt.database_transaction_log_bytes_used / 1048576.0 AS log_used_mb,
    st.session_id,
    s.login_name,
    s.host_name,
    s.program_name
FROM sys.dm_tran_database_transactions AS dt
JOIN sys.dm_tran_session_transactions AS st
    ON st.transaction_id = dt.transaction_id
JOIN sys.dm_tran_active_transactions AS at
    ON at.transaction_id = dt.transaction_id
JOIN sys.dm_exec_sessions AS s
    ON s.session_id = st.session_id
WHERE dt.database_id = DB_ID(N'tempdb')
ORDER BY at.transaction_begin_time;
