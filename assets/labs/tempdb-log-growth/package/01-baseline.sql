SET NOCOUNT ON;
SELECT
    @@SERVERNAME AS server_name,
    CAST(SERVERPROPERTY('Edition') AS nvarchar(128)) AS edition,
    CAST(SERVERPROPERTY('ProductVersion') AS nvarchar(32)) AS product_version,
    GETDATE() AS captured_at;
GO
USE tempdb;
GO
SELECT
    name, type_desc, physical_name,
    CAST(size * 8.0 / 1024 AS decimal(12,2)) AS size_mb,
    CASE WHEN is_percent_growth = 1 THEN CONCAT(growth, '%')
         ELSE CONCAT(CAST(growth * 8.0 / 1024 AS decimal(12,2)), ' MB')
    END AS filegrowth
FROM sys.database_files
ORDER BY file_id;
GO
SELECT
    total_log_size_in_bytes / 1048576.0 AS total_log_mb,
    used_log_space_in_bytes / 1048576.0 AS used_log_mb,
    used_log_space_in_percent
FROM sys.dm_db_log_space_usage;
GO
SELECT name, state_desc, recovery_model_desc, log_reuse_wait_desc
FROM sys.databases
WHERE database_id = DB_ID(N'tempdb');
GO