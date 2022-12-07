import unittest
from asr4.types.language import Language


class TestLanguage(unittest.TestCase):
    def testParse(self):
        self.assertEqual(Language.parse("en-us"), Language.EN_US)
        self.assertEqual(Language.parse("es"), Language.ES)
        self.assertEqual(Language.parse("pt-br"), Language.PT_BR)
        self.assertEqual(Language.parse("en-US"), Language.EN_US)
        self.assertEqual(Language.parse("es"), Language.ES)
        self.assertEqual(Language.parse("pt-BR"), Language.PT_BR)
        self.assertEqual(Language.parse("EN_US"), Language.EN_US)
        self.assertEqual(Language.parse("ES"), Language.ES)
        self.assertEqual(Language.parse("PT_BR"), Language.PT_BR)
        self.assertEqual(Language.parse("INVALID"), None)
        self.assertEqual(Language.parse(""), None)

    def testCheck(self):
        self.assertTrue(Language.check("en-us"))
        self.assertTrue(Language.check("es"))
        self.assertTrue(Language.check("pt-br"))
        self.assertTrue(Language.check("en-US"))
        self.assertTrue(Language.check("es"))
        self.assertTrue(Language.check("pt-BR"))
        self.assertTrue(Language.check("EN_US"))
        self.assertTrue(Language.check("ES"))
        self.assertTrue(Language.check("PT_BR"))
        self.assertFalse(Language.check("INVALID"))
        self.assertFalse(Language.check(""))

    def testDefault(self):
        self.assertEqual(Language.getDefaultValue(), Language.EN_US)

    def testAvailableOptions(self):
        self.assertEqual(
            Language.getValidOptions(), [Language.EN_US, Language.ES, Language.PT_BR]
        )
