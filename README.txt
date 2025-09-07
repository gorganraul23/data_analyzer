INFO

I. CONFIGURATION
1. Open sever/settings.py
    - search for DATABASES settings
    - be sure to have the correct configuration for your DB (name, user, password)

II. USE
1. Start the app
    - in terminal, go to the path where manage.py is
    - python manage.py runserver

2. Open on web
    - open http://127.0.0.1:8000/

3. See the 3 pages for data management. Start the operations from left to right.

4. Open student-activities page
    - here you can see the student's activities that are done and can be analyzed
    STEPS:
        - choose a student from DB
        - choose the 'done' activities
        - choose the activity types you want. It's a LIKE in SQL, comma separated (e.g. 'emotion,delay'), leave empty for all
        - choose the minimum count of valid IBI values (usually 120), -1 for no minimum limit
        - choose the rows count limit, -1 default (all)
        - search and see the results in the table
        - UNDER THE INPUTS, YOU HAVE THE ACTIVITY IDS AND YOU CAN COPY THEM FOR THE NEXT PAGE.

5. Open next page - Sensor Heart Rate
    - here you can see the table sensor_heart_rate based on activity ids
    STEPS:
        - PASTE the previous copied activity ids
        - set the limit to -1 to take all and click 'Search'
        - click 'Download CSV' to download a CSV with all the data from the table
        - CLICK 'DOWNLOAD PROCESSED CSV' TO DOWNLOAD THE CSV READY FOR ANALYSIS

6. Go to the third page - HRV Metrics
    - here you can upload the processed CSV and compute the HRV metrics.
    - the HRV metrics are computed using 2 methods: PyHRV library and manual written functions
    - there will be 2 tables with the results
    - when click on Compute and View: if Include PyHRV is checked, the first table will be competed,
        if Include Custom is checked, the second table will be competed,
        if both are checked, both tables will be completed.
    - when click on Compute and Download: if only Include Custom is checked, custom results are downloaded,
        else, PhHRV results are downloaded
    STEPS:
        - UPLOAD THE PREVIOUS GENERATED CSV
        - Compute and View - only to see the results
        - COMPUTE AND DOWNLOAD - to download the CSV with the results

7. Interpretation page
    - here you can see the differences between relax and cognitive tasks
    - with green - the transition is good (increase / decrease)
    - with red - the transition is not as expected
    STEPS:
        - Upload the previous CSV with the results
        - View - to only view the results again
        - View and Analyze - to start the analysis

8. Merge CSV page
    - here you can merge all HRV metrics CSVs into one to be used in the AI model or to analyze the data easier
    STEPS:
        - upload all CSVs you want to merge
        - download the merged CSV