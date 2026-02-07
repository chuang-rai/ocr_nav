import kuzu

kuzu_db = kuzu.Database("/tmp/kuzu_test_db")
connection = kuzu.Connection(kuzu_db)

connection.execute("CREATE (n:Object {id: 0, position: [1.0, 2.6, 2.1], color: 'red'});")
