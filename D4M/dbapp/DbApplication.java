package db;

import py4j.GatewayServer;

public class DbApplication {

    public static void main(String[] args) {
        DbApplication DB = new DbApplication();
        GatewayServer server = new GatewayServer(DB);
        
        // This will start the Py4J server and now, the JVM is ready to receive Python commands.
        // Once gateway.shutdown is called on the Python side, this call will return, and the Java program
        // will exit.
        // Obviously, this can be way more complex, but it's a good start :-)
        server.start();
        System.out.println("Starting server...");
    }

}