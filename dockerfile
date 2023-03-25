FROM alpine:3.16.4


#RUN apk add update && apk add upgrade
RUN apk add git openssh
ARG SSH_PRIVATE_KEY

RUN mkdir /root/.ssh/ && echo "${SSH_PRIVATE_KEY}" > /root/.ssh/id_rsa && chmod 600 /root/.ssh/id_rsa && ssh-keyscan github.com >> /root/.ssh/known_hosts
RUN git clone git@github.com:Sebastian1811/Procesamiento_DataSets.git

RUN git clone git@github.com:Sebastian1811/Procesamiento_DataSets.git
WORKDIR /app
ENTRYPOINT  ["sh"]