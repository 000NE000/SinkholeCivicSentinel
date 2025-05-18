cat /Users/macforhsj/Desktop/SinkholeCivicSentinel/db/schema.sql \
| docker exec -i sinkholecivicsentinel-db-1 \
      psql -U sinkhole_user -d sinkhole -f -