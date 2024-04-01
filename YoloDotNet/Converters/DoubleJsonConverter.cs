namespace YoloDotNet.Converters
{
    public class DoubleJsonConverter : JsonConverter<double>
    {
        public override double Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
            => double.Parse(reader.GetString()!, CultureInfo.InvariantCulture);

        public override void Write(Utf8JsonWriter writer, double value, JsonSerializerOptions options)
            => writer.WriteStringValue(value.ToString(CultureInfo.InvariantCulture));
    }
}
