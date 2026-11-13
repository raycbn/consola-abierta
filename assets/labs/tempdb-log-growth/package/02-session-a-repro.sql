/*
DISPOSABLE LAB ONLY.
Creates a dedicated table in tempdb, inserts 84 MB of payload,
holds the transaction open for three minutes, then rolls it back.
Do not run on production or a shared SQL Server instance.
*/
USE tempdb;
SET NOCOUNT ON;

IF OBJECT_ID(N'dbo.CA_TempdbLogGrowthLab', N'U') IS NOT NULL
    DROP TABLE dbo.CA_TempdbLogGrowthLab;

CREATE TABLE dbo.CA_TempdbLogGrowthLab
(
    id int IDENTITY(1,1) NOT NULL PRIMARY KEY,
    payload char(7000) NOT NULL
);

BEGIN TRANSACTION CA_TempdbLogGrowthLab;

;WITH n AS
(
    SELECT TOP (12000)
           ROW_NUMBER() OVER (ORDER BY (SELECT NULL)) AS n
    FROM sys.all_objects AS a
    CROSS JOIN sys.all_objects AS b
)
INSERT INTO dbo.CA_TempdbLogGrowthLab (payload)
SELECT REPLICATE('X', 7000)
FROM n;

SELECT
    @@SPID AS repro_session_id,
    COUNT_BIG(*) AS rows_inserted,
    SUM(DATALENGTH(payload)) AS payload_bytes,
    GETDATE() AS transaction_opened_at
FROM dbo.CA_TempdbLogGrowthLab;

RAISERROR('LAB TRANSACTION OPEN: run session B diagnostics now.', 10, 1) WITH NOWAIT;
WAITFOR DELAY '00:03:00';

ROLLBACK TRANSACTION CA_TempdbLogGrowthLab;
DROP TABLE dbo.CA_TempdbLogGrowthLab;
PRINT 'LAB COMPLETE: transaction rolled back and test table removed.';
