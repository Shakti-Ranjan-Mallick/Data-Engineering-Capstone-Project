rm -r /home/anabig114217/capstone_project
rm /home/anabig114217/mysql_tables.txt
rm /home/anabig114217/output_impala_analysis.txt
rm -r /home/anabig114217/Capstone_Outputs

mkdir Capstone_Outputs
mkdir capstone_project
hdfs dfs -mkdir capstoneproject

mysql -u anabig114217 -pBigdata123 -D anabig114212 -e 'source mysqlscript.sql' > /home/anabig114212/Capstone_Outputs/mysql_tables.txt

cp sqoopscript.sh capstone_project 
cd capstone_project
sh sqoopscript.sh

cd ..

hive -f hive_database.sql

impala-shell -i ip-10-1-2-103.ap-south-1.compute.internal -f impala_analysis_script.sql > /home/anabig114217/Capstone_Outputs/Cap_ImpalaAnalysis.txt

spark-submit Pyspark_analysis_sheet.py > /home/anabig114217/Capstone_Outputs/Output_Sparksql_eda_ml.txt
