using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace YoloDotNet.Configuration
{
    public static class FontConfig
    {
        public static int DefaultFontSize { get; set; } = 18;

        public static string DefaultFontFamilyName { get; set; } = SKTypeface.Default.FamilyName;

    }
}
