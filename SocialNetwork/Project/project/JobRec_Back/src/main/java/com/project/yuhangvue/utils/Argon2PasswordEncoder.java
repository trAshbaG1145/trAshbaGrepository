package com.project.yuhangvue.utils;/*
 *   @Author:田宇航
 *   @Date: 2025/1/7 13:58
 */

import de.mkammerer.argon2.Argon2;
import de.mkammerer.argon2.Argon2Factory;

public class Argon2PasswordEncoder {

    private static final int ITERATIONS = 2;
    private static final int MEMORY = 65536;
    private static final int PARALLELISM = 1;
    public static final Argon2Factory.Argon2Types TYPE = Argon2Factory.Argon2Types.ARGON2id;

    private static final Argon2 INSTANCE = Argon2Factory.create(TYPE);

    public static String encode(String password) {
        return INSTANCE.hash(ITERATIONS, MEMORY, PARALLELISM, password.toCharArray());
    }

    public static boolean matches(String encodedPassword, String password) {
        return INSTANCE.verify(encodedPassword, password.toCharArray());
    }

    public static void main(String[] args) {
        String password = "TYH041113";
        String hashedPassword = encode(password);
        System.out.println("Hashed Password: " + hashedPassword);
        System.out.println("Password matches: " + matches(hashedPassword, password));
    }
}


