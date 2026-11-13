CONSOLA ABIERTA — TEMPDB LOG GROWTH LAB
=========================================

PURPOSE
-------
A controlled reproduction of tempdb transaction-log growth caused by an
open user transaction. This is a learning lab, NOT a production runbook
and NOT a production incident.

TESTED ENVIRONMENT
------------------
Windows Server 2025 Standard Evaluation
SQL Server 2022 Developer RTM 16.0.1000.6
SSMS 22.12.0
Two tempdb data files and one log file.
For the measured run, the tempdb log started at 8 MB and used a fixed
8 MB autogrowth increment. That growth setting was intentional to expose
the events and is not a recommendation for production.

SAFETY
------
- Use a disposable, isolated lab instance only.
- Do not run the reproduction script on production or a shared instance.
- Script 02 inserts 12,000 rows (84,000,000 payload bytes) into tempdb
  and holds a transaction open for three minutes.
- Script 02 drops the dedicated lab table if it already exists.
- Do not run multiple copies of the reproduction at the same time.
- The package intentionally excludes the lab reset script that used
  DBCC SHRINKFILE. Shrinking is not the remediation demonstrated here.
- Do not use KILL on a session without identifying its owner and workload.
- Use a separate database connection for Session B diagnostics.

RUN ORDER
---------
1. Run 01-baseline.sql and record the initial file size, usage and
   log_reuse_wait_desc.
2. Open TWO SQL sessions connected to the same disposable SQL Server.
3. In Session A, run 02-session-a-repro.sql.
4. While Session A is in WAITFOR (three minutes), run
   03-session-b-diagnose.sql in Session B.
5. After Session A finishes its ROLLBACK and table cleanup, run
   04-after-rollback-verify.sql in either session.
6. Run 05-default-trace-autogrowth.sql promptly after the reproduction.
   It reads the recent 15-minute window and requires the default trace.
   The default trace may be unavailable or may not retain the events.

OBSERVED RESULTS IN THE RECORDED LAB
------------------------------------
Baseline: 8 MB log; approximately 5.13% used; reuse wait NOTHING.
During reproduction: log reached 216 MB; 99.08–99.16% used;
reuse wait ACTIVE_TRANSACTION.
Default trace: 26 log growths x 8 MB = 208 MB; two data-file growths
x 64 MB = 128 MB.
After rollback and CHECKPOINT: physical log remained 216 MB; 1.70–1.78%
used; reuse wait NOTHING; zero active user transactions in tempdb;
the dedicated test table was removed.

INTERPRETATION
--------------
The rollback made log space reusable again. It did not shrink the
physical log file. A 216 MB file with 1.78% used is different from a
216 MB file with 99% used. File shrink is not the corrective action
for the open transaction demonstrated here.

PERMISSIONS
-----------
The DMV queries require appropriate server-level performance state
permissions on SQL Server 2022 (for example VIEW SERVER PERFORMANCE
STATE, or sysadmin).

FILES
-----
01-baseline.sql
02-session-a-repro.sql
03-session-b-diagnose.sql
04-after-rollback-verify.sql
05-default-trace-autogrowth.sql

All scripts should be reviewed and adapted to the target lab before use.
