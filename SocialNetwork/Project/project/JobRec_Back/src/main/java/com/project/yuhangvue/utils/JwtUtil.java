package com.project.yuhangvue.utils;

import com.project.yuhangvue.entity.Candidate;
import com.project.yuhangvue.entity.Firm;
import com.project.yuhangvue.entity.Tokenizable;
import io.jsonwebtoken.*;
import org.springframework.stereotype.Component;

import javax.crypto.SecretKey;
import java.lang.reflect.Method;
import java.util.Date;
import java.util.HashMap;
import java.util.Map;
import java.util.UUID;

/**
 * @Author:田宇航
 * @Date: 2024/10/29 13:52
 */
@Component
public class JwtUtil {
    private static long time = 1000*120*60;
    private static String signature = "yuhang";

    public static <T extends Tokenizable> String createToken(T user) {
        JwtBuilder jwtBuilder = Jwts.builder();
        Map<String, Object> claims = new HashMap<>();

        // 使用反射获取用户信息
        try {
            Method getIdMethod = user.getClass().getMethod("getId");
            Method getNicknameMethod = user.getClass().getMethod("getNickname");
            Method getAvatarMethod = user.getClass().getMethod("getAvatar");

            claims.put("id", getIdMethod.invoke(user));
            claims.put("nickname", getNicknameMethod.invoke(user));
            claims.put("avatar", getAvatarMethod.invoke(user));
        } catch (Exception e) {
            throw new RuntimeException("Failed to create token", e);
        }

        String jwtToken = jwtBuilder
                .setHeaderParam("type", "JWT")
                .setHeaderParam("alg", "HS256")
                .setClaims(claims)
                .setSubject("admin-test")
                .setExpiration(new Date(System.currentTimeMillis() + time))
                .setId(UUID.randomUUID().toString())
                .signWith(SignatureAlgorithm.HS256, signature)
                .compact();

        return jwtToken;
    }

    public static boolean validateJwt(String token) {
        try {
            JwtParser parser = Jwts.parser();
            Jws<Claims> claimsJws = parser.setSigningKey(signature).parseClaimsJws(token);
            Claims claims = claimsJws.getBody();

            Date expiration = claims.getExpiration();
            if (expiration.before(new Date())) {
                System.out.println("Token已过期");
                return false;
            }
            return true;
        } catch (Exception e) {
            return false;
        }//设置需要解析的jwt
    }

    public static Claims verifyJwt(String token) {
        //签名秘钥，和生成的签名的秘钥一模一样
        Claims claims;
        try {
            claims = Jwts.parser()  //得到DefaultJwtParser
                    .setSigningKey(signature)         //设置签名的秘钥
                    .parseClaimsJws(token).getBody();

        } catch (Exception e) {
            claims = null;
        }//设置需要解析的jwt
        return claims;
    }

}

