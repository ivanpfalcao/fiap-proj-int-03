#!/bin/bash

# Get current file location
BASEDIR=$(cd -P -- "$(dirname -- "${0}")" && pwd -P)
NAMESPACE="trino"

S3_ACCESS_KEY=UoFasnX07FYhrFrSU9EJ
S3_SECRET_KEY=XUbweXDQFchZC49AFADLBgZ3G9urc8SSvRURIP4m
S3_ENDPOINT=minio-service.trino.svc.cluster.local:9000
DB_CONNECTION="jdbc:postgresql://hive-metastore-postgresql.${NAMESPACE}.svc.cluster.local:5432/metastore"
DB_USER="hive"
DB_PSWD="hive-password"
#HIVE_METASTORE_URIS="thrift://hive-metastore-0.hive-metastore.${NAMESPACE}.svc.cluster.local:9083"
HIVE_METASTORE_URIS="thrift://hive-metastore-service:9083"
SERVICE_OPTS="-Dhive.metastore.uris=${HIVE_METASTORE_URIS} -Djavax.jdo.option.ConnectionDriverName=org.postgresql.Driver -Djavax.jdo.option.ConnectionURL=${DB_CONNECTION} -Djavax.jdo.option.ConnectionUserName=${DB_USER} -Djavax.jdo.option.ConnectionPassword=${DB_PSWD}"

kubectl -n ${NAMESPACE} delete secret hive-meta-conn-secret
kubectl -n ${NAMESPACE} create secret generic hive-meta-conn-secret \
    --from-literal=SERVICE_OPTS="$SERVICE_OPTS" \
    --from-literal=DB_CONNECTION="$DB_CONNECTION" \
    --from-literal=DB_USER="$DB_USER" \
    --from-literal=DB_PSWD="$DB_PSWD"


xml_content=$(cat <<EOF
<?xml version="1.0" encoding="UTF-8"?>
<configuration>
    <property>
        <name>hive.server2.enable.doAs</name>
        <value>false</value>
    </property>
    <property>
        <name>hive.tez.exec.inplace.progress</name>
        <value>false</value>
    </property>
    <property>
        <name>hive.exec.scratchdir</name>
        <value>/opt/hive/scratch_dir</value>
    </property>
    <property>
        <name>hive.user.install.directory</name>
        <value>/opt/hive/install_dir</value>
    </property>
    <property>
        <name>tez.runtime.optimize.local.fetch</name>
        <value>true</value>
    </property>
    <property>
        <name>hive.exec.submit.local.task.via.child</name>
        <value>false</value>
    </property>
    <property>
        <name>mapreduce.framework.name</name>
        <value>local</value>
    </property>
    <property>
        <name>tez.local.mode</name>
        <value>true</value>
    </property>
    <property>
        <name>hive.execution.engine</name>
        <value>mr</value>
    </property>
    <property>
        <name>metastore.warehouse.dir</name>
        <value>/opt/hive/data/warehouse</value>
    </property>
    <property>
        <name>metastore.metastore.event.db.notification.api.auth</name>
        <value>false</value>
    </property>


    <property>
        <name>fs.s3a.access.key</name>
        <value>${S3_ACCESS_KEY}</value>
    </property>
    <property>
        <name>fs.s3a.secret.key</name>
        <value>${S3_SECRET_KEY}</value>
    </property>
    <property>
        <name>fs.s3a.connection.ssl.enabled</name>
        <value>false</value>
    </property>
    <property>
        <name>fs.s3a.path.style.access</name>
        <value>true</value>
    </property>
    <property>
        <name>fs.s3a.endpoint</name>
        <value>${S3_ENDPOINT}</value>
    </property>    
    <property>
        <name>javax.jdo.option.ConnectionDriverName</name>
        <value>org.postgresql.Driver</value>
    </property>
    <property>
        <name>hive.metastore.uris</name>
        <value>${HIVE_METASTORE_URIS}</value>
    </property>    
    <property>
        <name>javax.jdo.option.ConnectionURL</name>
        <value>${DB_CONNECTION}</value>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionUserName</name>
        <value>${DB_USER}</value>
    </property>
    <property>
        <name>javax.jdo.option.ConnectionPassword</name>
        <value>${DB_PSWD}</value>
    </property>
</configuration>
EOF
)

# Create Kubernetes secret
kubectl -n ${NAMESPACE} delete secret hive-site-xml
kubectl -n ${NAMESPACE} create secret generic hive-site-xml \
    --from-literal=hive-site.xml="$xml_content"