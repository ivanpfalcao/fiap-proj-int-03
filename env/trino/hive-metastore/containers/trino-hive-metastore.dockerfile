FROM apache/hive:3.1.3

USER root

RUN apt update \
    && apt install -y \
    wget && \
    htop && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

WORKDIR /opt/hive/lib

RUN wget https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.11.1026/aws-java-sdk-bundle-1.11.1026.jar \
    && wget https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.2.2/hadoop-aws-3.2.2.jar

WORKDIR /opt/hive

RUN chmod 777 /opt/hive/conf/hive-site.xml

USER hive

ENTRYPOINT ["sh", "-c", "cp ${HIVE_CUSTOM_CONF_DIR}/hive-site.xml -f /opt/hive/conf/hive-site.xml && /entrypoint.sh"]
