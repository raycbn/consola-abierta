SET NOCOUNT ON;

DECLARE @trace nvarchar(4000);
SELECT @trace = path
FROM sys.traces
WHERE is_default = 1;

IF @trace IS NULL
BEGIN
    RAISERROR('The default trace is not enabled or available on this instance.', 10, 1);
    RETURN;
END;

SELECT
    EventClass,
    CASE EventClass
        WHEN 93 THEN 'Log File Auto Grow'
        WHEN 92 THEN 'Data File Auto Grow'
    END AS event_name,
    COUNT_BIG(*) AS growth_events,
    SUM(IntegerData) * 8.0 / 1024 AS total_growth_mb,
    MIN(StartTime) AS first_event,
    MAX(StartTime) AS last_event
FROM sys.fn_trace_gettable(@trace, DEFAULT)
WHERE EventClass IN (92,93)
  AND DatabaseName = N'tempdb'
  AND StartTime >= DATEADD(MINUTE, -15, GETDATE())
GROUP BY EventClass
ORDER BY EventClass;
