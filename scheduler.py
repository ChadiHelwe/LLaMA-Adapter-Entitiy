from apscheduler.schedulers.blocking import BlockingScheduler
import os

def cron_kinit():
    print("kinit -R")
    os.system("kinit -R")

if __name__ == "__main__":
    print("Cron Scheduler")
    scheduler = BlockingScheduler()
    scheduler.add_job(cron_kinit, "interval", hours=8)
    scheduler.start()

