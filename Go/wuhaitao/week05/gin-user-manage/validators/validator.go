package validators

import (
	"reflect"
	"strings"

	"github.com/gin-gonic/gin/binding"
	"github.com/go-playground/locales/zh"
	ut "github.com/go-playground/universal-translator"
	"github.com/go-playground/validator/v10"
	zh_translations "github.com/go-playground/validator/v10/translations/zh"
)

// 全局翻译器
var (
	uni   *ut.UniversalTranslator
	trans ut.Translator
)

// 初始化验证器
func InitValidator() {
	// 获取gin的验证器
	if v, ok := binding.Validator.Engine().(*validator.Validate); ok {
		// 注册一个获取json tag的自定义方法
		v.RegisterTagNameFunc(func(fld reflect.StructField) string {
			name := strings.SplitN(fld.Tag.Get("json"), ",", 2)[0]
			if name == "-" {
				return ""
			}
			return name
		})

		// 创建翻译器
		zhTrans := zh.New()
		uni = ut.New(zhTrans, zhTrans)

		// 获取中文翻译器
		trans, _ = uni.GetTranslator("zh")

		// 注册翻译器
		zh_translations.RegisterDefaultTranslations(v, trans)

		// 添加自定义翻译
		v.RegisterTranslation("email", trans, func(ut ut.Translator) error {
			return ut.Add("email", "{0}格式不正确", true)
		}, func(ut ut.Translator, fe validator.FieldError) string {
			t, _ := ut.T("email", fe.Field())
			return t
		})
	}
}

// 翻译错误信息
func TranslateError(err error) string {
	if validationErrors, ok := err.(validator.ValidationErrors); ok {
		for _, e := range validationErrors {
			return e.Translate(trans)
		}
	}
	return err.Error()
}
