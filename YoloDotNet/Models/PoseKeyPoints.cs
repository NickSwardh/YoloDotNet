// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: 2026 Niklas Swärd
// https://github.com/NickSwardh/YoloDotNet

namespace YoloDotNet.Models
{
    /// <summary>
    /// Describes how the keypoints of a pose model are colored and connected when drawing pose estimation results.
    /// Load one from a JSON file with <see cref="Load(string)"/>.
    /// </summary>
    /// <remarks>
    /// <para>JSON format:</para>
    /// <code>
    /// {
    ///   "name": "left-arm",
    ///   "colors": { "arm": "#33ACFF", "torso": "#FF33AC" },
    ///   "keypoints": [
    ///     { "name": "left_shoulder", "color": "arm" },
    ///     { "name": "left_elbow",    "color": "arm" },
    ///     { "name": "left_wrist",    "color": "arm" },
    ///     { "name": "left_hip" }
    ///   ],
    ///   "connections": [
    ///     { "from": "left_shoulder", "to": "left_elbow" },
    ///     { "from": "left_elbow",    "to": "left_wrist" },
    ///     { "from": "left_shoulder", "to": "left_hip", "color": "torso" }
    ///   ]
    /// }
    /// </code>
    /// <list type="bullet">
    /// <item>Keypoints must be listed in the SAME order as the model outputs them. A count can be checked with
    /// <see cref="ValidateKeyPointCount"/>, but the order cannot be checked, so get it right.</item>
    /// <item>Each keypoint needs a unique name (case-sensitive). Connections refer to keypoints by name.</item>
    /// <item>A color is either a key from <c>colors</c> or a literal <c>#RRGGBB</c>. Opacity comes from
    /// <see cref="CommonBoundingBoxOptions.BoundingBoxOpacity"/>.</item>
    /// <item>A keypoint without a color is drawn with <see cref="PoseDrawingOptions.DefaultPoseColor"/>, which keeps
    /// its own alpha (<see cref="CommonBoundingBoxOptions.BoundingBoxOpacity"/> is not applied to it).
    /// A connection without a color uses the color of its <c>from</c> keypoint.</item>
    /// <item><c>name</c> and <c>colors</c> are optional. Comments and trailing commas are allowed;
    /// unknown and repeated properties are errors.</item>
    /// </list>
    /// </remarks>
    public sealed class PoseKeyPoints
    {
        private readonly PoseKeyPointsDefinition _definition;

        /// <summary>
        /// Optional name of the pose keypoints.
        /// </summary>
        public string? Name { get; }

        /// <summary>
        /// Keypoints in model output order.
        /// </summary>
        public IReadOnlyList<PoseKeyPoint> KeyPoints { get; }

        /// <summary>
        /// Connections between keypoints, in the order they were defined.
        /// </summary>
        public IReadOnlyList<PoseKeyPointConnection> Connections { get; }

        /// <summary>
        /// Number of keypoints defined.
        /// </summary>
        public int KeyPointCount => KeyPoints.Count;

        /// <summary>
        /// Keypoint colors by keypoint index. Null means <see cref="PoseDrawingOptions.DefaultPoseColor"/>.
        /// </summary>
        internal SKColor?[] KeyPointColors { get; }

        /// <summary>
        /// Outgoing connections grouped by their <c>from</c> keypoint index, in definition order.
        /// </summary>
        internal (int To, SKColor? Color)[][] ConnectionsByKeyPoint { get; }

        private PoseKeyPoints(PoseKeyPointsDefinition definition)
        {
            _definition = definition;
            Name = definition.Name;

            var keyPointDefinitions = definition.KeyPoints
                ?? throw Invalid("keypoints", "is missing. Define at least one keypoint.");

            if (keyPointDefinitions.Count == 0)
                throw Invalid("keypoints", "is empty. Define at least one keypoint.");

            var palette = ResolvePalette(definition.Colors);

            var indexByName = new Dictionary<string, int>(StringComparer.Ordinal);
            var keyPoints = new PoseKeyPoint[keyPointDefinitions.Count];

            for (int i = 0; i < keyPointDefinitions.Count; i++)
            {
                var path = $"keypoints[{i}]";
                var keyPoint = keyPointDefinitions[i] ?? throw Invalid(path, "is null.");

                if (string.IsNullOrWhiteSpace(keyPoint.Name))
                    throw Invalid($"{path}.name", "is missing. Every keypoint needs a unique name.");

                if (indexByName.TryAdd(keyPoint.Name, i) is false)
                    throw Invalid($"{path}.name", $"\"{keyPoint.Name}\" is already used by keypoints[{indexByName[keyPoint.Name]}].");

                keyPoints[i] = new PoseKeyPoint(keyPoint.Name, ResolveColor(keyPoint.Color, palette, $"{path}.color"));
            }

            var connectionDefinitions = definition.Connections ?? [];
            var connections = new PoseKeyPointConnection[connectionDefinitions.Count];

            for (int i = 0; i < connectionDefinitions.Count; i++)
            {
                var path = $"connections[{i}]";
                var connection = connectionDefinitions[i] ?? throw Invalid(path, "is null.");

                var from = ResolveKeyPoint(connection.From, indexByName, $"{path}.from");
                var to = ResolveKeyPoint(connection.To, indexByName, $"{path}.to");

                if (from == to)
                    throw Invalid(path, $"connects \"{connection.From}\" to itself.");

                var color = ResolveColor(connection.Color, palette, $"{path}.color") ?? keyPoints[from].Color;

                connections[i] = new PoseKeyPointConnection(from, to, color);
            }

            KeyPoints = Array.AsReadOnly(keyPoints);
            Connections = Array.AsReadOnly(connections);
            KeyPointColors = [.. keyPoints.Select(k => k.Color)];
            ConnectionsByKeyPoint = [.. Enumerable.Range(0, keyPoints.Length)
                .Select(i => connections.Where(c => c.From == i).Select(c => (c.To, c.Color)).ToArray())];
        }

        #region Loading
        /// <summary>
        /// Loads pose keypoints from a JSON file.
        /// </summary>
        /// <exception cref="InvalidDataException">The JSON is malformed or the pose keypoint definition is invalid.</exception>
        public static PoseKeyPoints Load(string path)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(path);

            using var stream = File.OpenRead(path);
            return Load(stream);
        }

        /// <summary>
        /// Loads pose keypoints from a JSON string.
        /// </summary>
        /// <exception cref="InvalidDataException">The JSON is malformed or the pose keypoint definition is invalid.</exception>
        public static PoseKeyPoints Parse(string json)
        {
            ArgumentNullException.ThrowIfNull(json);

            return ParseUtf8(Encoding.UTF8.GetBytes(json));
        }

        /// <summary>
        /// Loads pose keypoints from a stream containing JSON. The stream is read to the end and left open.
        /// </summary>
        /// <exception cref="InvalidDataException">The JSON is malformed or the pose keypoint definition is invalid.</exception>
        public static PoseKeyPoints Load(Stream stream)
        {
            ArgumentNullException.ThrowIfNull(stream);

            using var buffer = new MemoryStream();
            stream.CopyTo(buffer);

            return ParseUtf8(buffer.ToArray());
        }

        /// <summary>
        /// Exports the pose keypoints as JSON in the same format it is loaded from. Use it to get a template, e.g.
        /// <c>File.WriteAllText("my-pose-keypoints.json", poseKeyPoints.ToJson())</c>.
        /// Comments and formatting of the original file are not preserved.
        /// </summary>
        public string ToJson()
            => JsonSerializer.Serialize(_definition, PoseKeyPointsJsonContext.Default.PoseKeyPointsDefinition);

        private static PoseKeyPoints ParseUtf8(byte[] utf8Json)
        {
            ReadOnlySpan<byte> json = utf8Json;

            // Files saved by some editors start with a UTF-8 byte order mark
            if (json.StartsWith(Encoding.UTF8.Preamble))
                json = json[Encoding.UTF8.Preamble.Length..];

            PoseKeyPointsDefinition? definition;

            try
            {
                RejectDuplicateProperties(json);
                definition = JsonSerializer.Deserialize(json, PoseKeyPointsJsonContext.Default.PoseKeyPointsDefinition);
            }
            catch (JsonException ex)
            {
                throw new InvalidDataException($"Invalid pose keypoints JSON: {ex.Message}", ex);
            }

            return new PoseKeyPoints(definition ?? throw Invalid("$", "is null."));
        }

        /// <summary>
        /// The serializer silently keeps the last of two properties with the same name (e.g. a repeated "from"),
        /// so repeated properties are rejected up front.
        /// </summary>
        private static void RejectDuplicateProperties(ReadOnlySpan<byte> json)
        {
            var reader = new Utf8JsonReader(json, new JsonReaderOptions
            {
                CommentHandling = JsonCommentHandling.Skip,
                AllowTrailingCommas = true
            });

            // Open containers, innermost on top. Names is null for arrays.
            var containers = new Stack<JsonContainer>();
            string? property = null;

            while (reader.Read())
            {
                if (reader.TokenType == JsonTokenType.PropertyName)
                {
                    var owner = containers.Peek();
                    property = reader.GetString()!;

                    if (owner.Names!.Add(property) is false)
                        throw Invalid(owner.Path, $"has the property \"{property}\" more than once.");

                    continue;
                }

                if (reader.TokenType is JsonTokenType.EndObject or JsonTokenType.EndArray)
                {
                    containers.Pop();
                    continue;
                }

                // A value: the root, a property value, or an array element
                var path = containers.TryPeek(out var parent)
                    ? parent.Names is null ? $"{parent.Path}[{parent.NextIndex++}]" : PropertyPath(parent.Path, property!)
                    : "$";

                if (reader.TokenType is JsonTokenType.StartObject)
                    containers.Push(new JsonContainer(path, new HashSet<string>(StringComparer.Ordinal)));
                else if (reader.TokenType is JsonTokenType.StartArray)
                    containers.Push(new JsonContainer(path, null));

                property = null;
            }
        }

        private static string PropertyPath(string parentPath, string property)
        {
            var name = property.Length == 0 || property.AsSpan().ContainsAnyExcept(_plainPropertyChars)
                ? $"[{JsonSerializer.Serialize(property, PoseKeyPointsJsonContext.Default.String)}]"
                : property;

            return parentPath == "$"
                ? name.StartsWith('[') ? $"${name}" : name
                : name.StartsWith('[') ? $"{parentPath}{name}" : $"{parentPath}.{name}";
        }

        private static readonly SearchValues<char> _plainPropertyChars =
            SearchValues.Create("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-");

        private sealed class JsonContainer(string path, HashSet<string>? names)
        {
            public string Path { get; } = path;
            public HashSet<string>? Names { get; } = names;
            public int NextIndex { get; set; }
        }
        #endregion

        #region Validation
        /// <summary>
        /// Throws if the pose keypoints do not define exactly <paramref name="expectedKeyPointCount"/> keypoints,
        /// e.g. the keypoint count of the model it is used with.
        /// A matching count does not prove the keypoints are in the model's order.
        /// </summary>
        /// <exception cref="ArgumentException">The keypoint count does not match.</exception>
        public void ValidateKeyPointCount(int expectedKeyPointCount)
        {
            if (KeyPointCount != expectedKeyPointCount)
            {
                throw new ArgumentException(
                    $"Pose keypoints{(Name is null ? "" : $" '{Name}'")} defines {KeyPointCount} keypoints, but {expectedKeyPointCount} were expected.",
                    nameof(expectedKeyPointCount));
            }
        }

        private static Dictionary<string, SKColor> ResolvePalette(Dictionary<string, string>? colors)
        {
            var palette = new Dictionary<string, SKColor>(StringComparer.Ordinal);

            foreach (var (key, value) in colors ?? [])
            {
                var path = $"colors.{key}";

                if (key.StartsWith('#'))
                    throw Invalid(path, "color names must not start with '#'.");

                palette[key] = ParseHex(value, path);
            }

            return palette;
        }

        private static SKColor? ResolveColor(string? color, Dictionary<string, SKColor> palette, string path)
        {
            if (color is null)
                return null;

            if (color.StartsWith('#'))
                return ParseHex(color, path);

            return palette.TryGetValue(color, out var paletteColor)
                ? paletteColor
                : throw Invalid(path, $"\"{color}\" is neither a key in \"colors\" nor a #RRGGBB color.");
        }

        private static SKColor ParseHex(string? hex, string path)
        {
            if (hex is not { Length: 7 } || hex[0] != '#' || hex.AsSpan(1).ContainsAnyExcept(_hexDigits))
                throw Invalid(path, $"\"{hex}\" is not a #RRGGBB color.");

            var rgb = uint.Parse(hex.AsSpan(1), NumberStyles.HexNumber, CultureInfo.InvariantCulture);

            return new SKColor((byte)(rgb >> 16), (byte)(rgb >> 8), (byte)rgb);
        }

        private static readonly SearchValues<char> _hexDigits = SearchValues.Create("0123456789abcdefABCDEF");

        private static int ResolveKeyPoint(string? name, Dictionary<string, int> indexByName, string path)
        {
            if (string.IsNullOrWhiteSpace(name))
                throw Invalid(path, "is missing.");

            return indexByName.TryGetValue(name, out var index)
                ? index
                : throw Invalid(path, $"unknown keypoint \"{name}\".");
        }

        private static InvalidDataException Invalid(string path, string message)
            => new($"Invalid pose keypoints: {path} {message}");
        #endregion
    }

    /// <summary>
    /// A keypoint of a <see cref="PoseKeyPoints"/>.
    /// </summary>
    /// <param name="Name">Unique name of the keypoint.</param>
    /// <param name="Color">Keypoint color, or null to use <see cref="PoseDrawingOptions.DefaultPoseColor"/>.</param>
    public sealed record PoseKeyPoint(string Name, SKColor? Color);

    /// <summary>
    /// A connection between two keypoints of a <see cref="PoseKeyPoints"/>.
    /// </summary>
    /// <param name="From">Index of the keypoint the line starts at.</param>
    /// <param name="To">Index of the keypoint the line ends at.</param>
    /// <param name="Color">Line color, or null to use <see cref="PoseDrawingOptions.DefaultPoseColor"/>.</param>
    public sealed record PoseKeyPointConnection(int From, int To, SKColor? Color);

    #region JSON definition
    internal sealed class PoseKeyPointsDefinition
    {
        public string? Name { get; set; }
        public Dictionary<string, string>? Colors { get; set; }
        [JsonPropertyName("keypoints")]
        public List<PoseKeyPointDefinition?>? KeyPoints { get; set; }
        public List<PoseKeyPointConnectionDefinition?>? Connections { get; set; }
    }

    internal sealed class PoseKeyPointDefinition
    {
        public string? Name { get; set; }
        public string? Color { get; set; }
    }

    internal sealed class PoseKeyPointConnectionDefinition
    {
        public string? From { get; set; }
        public string? To { get; set; }
        public string? Color { get; set; }
    }

    [JsonSourceGenerationOptions(
        PropertyNamingPolicy = JsonKnownNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        UnmappedMemberHandling = JsonUnmappedMemberHandling.Disallow,
        ReadCommentHandling = JsonCommentHandling.Skip,
        AllowTrailingCommas = true,
        WriteIndented = true)]
    [JsonSerializable(typeof(PoseKeyPointsDefinition))]
    [JsonSerializable(typeof(string))]
    internal partial class PoseKeyPointsJsonContext : JsonSerializerContext
    {
    }
    #endregion
}
